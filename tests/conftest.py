"""Pytest fixtures for MLIR EDSL test suite

Provides:
    backend: Session-scoped C++ backend instance
    clean_module: Auto-clears module before each test, saves IR after (SAVE_IR=1)
    check_ir: FileCheck-based IR assertion fixture
"""

import os
import shutil
import subprocess
import tempfile

import pytest

from mlir_edsl.backend import get_backend


# ==================== HELPERS ====================

def _read_file(path):
    """Read file contents, return placeholder if file doesn't exist."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"(File not found: {path})"


def _generate_ir_html(test_name, func_name, func_source, ast_dump, mlir,
                      snapshots, unopt_llvm, opt_llvm,
                      failed: bool = False, failure_ir: str = ""):
    """Generate HTML with vertical pipeline layout showing all compilation stages."""
    from html import escape
    import datetime

    # Build lowering pipeline section
    pipeline_html = ""
    if snapshots:
        prev_ir = mlir
        for i, (pass_name, ir) in enumerate(snapshots):
            is_failure = pass_name.startswith("[FAILED]")
            changed = (ir != prev_ir)
            badge = "" if changed else ' <span class="badge-unchanged">(unchanged)</span>'
            # Auto-expand passes that changed IR and involve bufferization, or failures
            is_open = is_failure or (changed and "bufferize" in pass_name.lower())
            open_attr = " open" if is_open else ""
            if is_failure:
                css_class = "pass-failed"
            elif changed:
                css_class = "pass-changed"
            else:
                css_class = "pass-unchanged"
            escaped_name = escape(pass_name)
            escaped_ir = escape(ir)
            pipeline_html += (
                f'\n        <details{open_attr} class="{css_class}">'
                f"\n            <summary>Pass {i+1}: {escaped_name}{badge}</summary>"
                f'\n            <pre class="mlir">{escaped_ir}</pre>'
                f"\n        </details>"
            )
            prev_ir = ir
    else:
        pipeline_html = (
            '\n        <p style="color: #808080; font-style: italic;">'
            "No lowering snapshots captured (set SAVE_IR=1)</p>"
        )

    # Build failure banner and failure IR section
    title_prefix = "[FAILED] " if failed else ""
    title_color = "#f44747" if failed else "#4ec9b0"
    failure_banner_html = ""
    if failed:
        failure_banner_html = """
    <div style="background-color: #5a1d1d; border: 2px solid #f44747; border-radius: 5px; padding: 15px; margin-bottom: 20px;">
        <span style="color: #f44747; font-size: 20px; font-weight: bold;">&#x2717; TEST FAILED</span>
        <span style="color: #d4d4d4; margin-left: 10px;">Lowering pipeline failed — see Failure-Point IR below</span>
    </div>"""

    failure_ir_section_html = ""
    if failure_ir:
        failure_ir_section_html = f"""
    <!-- Failure-Point IR -->
    <h2 class="section-header" style="color: #f44747;">Failure-Point IR</h2>
    <details open>
        <summary style="color: #f44747;">Partially-Lowered IR at Failure</summary>
        <pre class="mlir">{escape(failure_ir)}</pre>
    </details>"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{escape(title_prefix)}IR Output: {escape(test_name)}</title>
    <style>
        body {{
            font-family: 'Consolas', 'Monaco', monospace;
            margin: 20px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-size: 16px;
        }}
        h1 {{
            color: {title_color};
            border-bottom: 2px solid {title_color};
            padding-bottom: 10px;
            font-size: 28px;
        }}
        .metadata {{
            color: #6a9955;
            font-style: italic;
            margin-bottom: 20px;
            font-size: 14px;
        }}
        .section-header {{
            color: #569cd6;
            font-size: 20px;
            margin: 30px 0 10px 0;
            padding-bottom: 5px;
            border-bottom: 1px solid #3c3c3c;
        }}
        .ir-row {{
            display: flex;
            gap: 10px;
            margin: 20px 0;
        }}
        .ir-column {{
            flex: 1;
            min-width: 0;
        }}
        details {{
            border: 1px solid #3c3c3c;
            border-radius: 5px;
            background-color: #252526;
            margin-bottom: 8px;
        }}
        .ir-column details {{
            height: 100%;
            margin-bottom: 0;
        }}
        summary {{
            padding: 15px;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            color: #4ec9b0;
            user-select: none;
        }}
        summary:hover {{
            background-color: #2d2d30;
        }}
        pre {{
            margin: 0;
            padding: 20px;
            overflow-x: auto;
            background-color: #1e1e1e;
            border-top: 1px solid #3c3c3c;
            font-size: 14px;
            line-height: 1.4;
        }}
        .python {{ color: #dcdcaa; }}
        .ast {{ color: #b5cea8; }}
        .mlir {{ color: #ce9178; }}
        .llvm-unopt {{ color: #dcdcaa; }}
        .llvm-opt {{ color: #9cdcfe; }}
        .badge-unchanged {{
            font-size: 12px;
            color: #6a9955;
            font-weight: normal;
        }}
        .pass-unchanged summary {{
            color: #808080;
        }}
        .pass-changed summary {{
            color: #4ec9b0;
        }}
        .pass-failed summary {{
            color: #f44747;
        }}
        .pass-failed {{
            border-color: #f44747;
        }}
        .pipeline-section {{
            margin: 10px 0 10px 20px;
        }}
    </style>
</head>
<body>
    <h1>{escape(title_prefix)}Test: {escape(test_name)}</h1>
    <div class="metadata">
        Function: {escape(func_name)}<br>
        Generated: {escape(str(datetime.datetime.now()))}
    </div>
{failure_banner_html}
    <!-- Python Source -->
    <h2 class="section-header">Python Source</h2>
    <details open>
        <summary>@ml_function</summary>
        <pre class="python">{escape(func_source) if func_source else "(no source captured)"}</pre>
    </details>

    <!-- AST -->
    <h2 class="section-header">AST</h2>
    <details open>
        <summary>Expression Tree</summary>
        <pre class="ast">{escape(ast_dump) if ast_dump else "(no AST dump captured)"}</pre>
    </details>

{failure_ir_section_html}
    <!-- Input MLIR -->
    <h2 class="section-header">Input MLIR</h2>
    <details open>
        <summary>MLIR Dialect</summary>
        <pre class="mlir">{escape(mlir)}</pre>
    </details>

    <!-- Lowering Pipeline -->
    <h2 class="section-header">Lowering Pipeline ({len(snapshots)} passes)</h2>
    <div class="pipeline-section">{pipeline_html}
    </div>

    <!-- LLVM IR (side-by-side) -->
    <h2 class="section-header">LLVM IR</h2>
    <div class="ir-row">
        <div class="ir-column">
            <details open>
                <summary>Unoptimized</summary>
                <pre class="llvm-unopt">{escape(unopt_llvm)}</pre>
            </details>
        </div>
        <div class="ir-column">
            <details open>
                <summary>Optimized (O2)</summary>
                <pre class="llvm-opt">{escape(opt_llvm)}</pre>
            </details>
        </div>
    </div>
</body>
</html>"""


def _save_ir_for_test(backend, node, failed: bool = False, failure_ir: str = ""):
    """Save IR files and generate HTML report for a test.

    Uses the pytest node ID to create a hierarchical output structure:
        tests/memref/test_array_execution.py::TestArrayExecution::test_basic
        -> ir_html/memref/test_array_execution/test_basic.html
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ir_output_dir = os.path.join(project_root, "ir_output")
    ir_html_dir = os.path.join(project_root, "ir_html")

    os.makedirs(ir_output_dir, exist_ok=True)

    functions = backend.list_functions()
    func_name = functions[0] if functions else "__unknown__"

    # Build hierarchical output path from node ID
    # e.g. "tests/memref/test_array_execution.py::TestClass::test_name"
    # ->   "memref/test_array_execution/test_name.html"
    node_id = node.nodeid
    # Strip "tests/" prefix
    rel = node_id.split("tests/", 1)[-1] if "tests/" in node_id else node_id
    # Split file path and test path
    file_part, _, test_part = rel.partition("::")
    # Remove .py extension, use as directory
    file_dir = file_part.replace(".py", "")
    # Use only the test method name (last segment after ::)
    test_name = test_part.split("::")[-1] if test_part else node.name

    html_dir = os.path.join(ir_html_dir, file_dir)
    os.makedirs(html_dir, exist_ok=True)

    # Read IR files written by C++ backend
    mlir_ir = _read_file(os.path.join(ir_output_dir, f"{func_name}.mlir"))
    unopt_ir = _read_file(os.path.join(ir_output_dir, "module_unopt.ll"))
    opt_ir = _read_file(os.path.join(ir_output_dir, "module_opt.ll"))

    # Get lowering pipeline snapshots, AST dump, and Python source
    snapshots = backend.get_lowering_snapshots()
    ast_dump = backend._ast_dumps.get(func_name, "")
    func_source = backend._func_sources.get(func_name, "")

    # Generate HTML report
    html_path = os.path.join(html_dir, f"{test_name}.html")
    html_content = _generate_ir_html(
        test_name, func_name, func_source, ast_dump, mlir_ir,
        snapshots, unopt_ir, opt_ir,
        failed=failed, failure_ir=failure_ir
    )
    with open(html_path, "w") as f:
        f.write(html_content)


def _find_filecheck():
    """Discover FileCheck binary."""
    # Environment variable override
    env_path = os.getenv("FILECHECK_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # PATH lookup
    found = shutil.which("FileCheck")
    if found:
        return found

    # Known LLVM build location
    known = os.path.expanduser("~/dev/llvm-project/build/bin/FileCheck")
    if os.path.isfile(known):
        return known

    return None


# ==================== HOOKS ====================

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach the call-phase result to the item so clean_module can read it."""
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        item.rep_call = rep


# ==================== FIXTURES ====================

@pytest.fixture(scope="session", autouse=True)
def session_setup():
    """Clear and create ir_output/ before any test runs.

    C++ saves {name}.mlir and module_unopt.ll during test execution.
    ir_output/ must exist before tests start, otherwise the C++ saves fail
    silently and the HTML report shows '(File not found: ...)'.
    """
    save_ir = os.getenv("SAVE_IR", "").lower() in ("1", "true", "yes")
    if save_ir:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ir_output_dir = os.path.join(project_root, "ir_output")
        ir_html_dir = os.path.join(project_root, "ir_html")
        if os.path.exists(ir_output_dir):
            shutil.rmtree(ir_output_dir)
        if os.path.exists(ir_html_dir):
            shutil.rmtree(ir_html_dir)
        os.makedirs(ir_output_dir)


@pytest.fixture(scope="session")
def backend():
    """Session-scoped C++ backend instance.

    Skips the test if the C++ backend is not available.
    """
    b = get_backend()
    if b is None:
        pytest.skip("C++ backend not available")
    return b


@pytest.fixture(autouse=True)
def clean_module(request):
    """Auto-clear module before each test, save IR after if SAVE_IR=1.

    Calls get_backend() directly (not via backend fixture) so that
    pure-Python AST tests are not skipped when the backend is unavailable.
    """
    b = get_backend()
    if b is not None:
        b.clear_module()

    yield

    if b is not None:
        save_ir = os.getenv("SAVE_IR", "").lower() in ("1", "true", "yes")
        rep = getattr(request.node, 'rep_call', None)
        test_failed = rep is not None and rep.failed
        if save_ir or test_failed:
            failure_ir = b.get_failure_ir() if test_failed else ""
            try:
                _save_ir_for_test(b, request.node, failed=test_failed,
                                  failure_ir=failure_ir)
            except Exception as e:
                print(f"\nWarning: Could not save IR: {e}")


@pytest.fixture
def check_ir(backend):
    """FileCheck-based IR assertion fixture.

    Usage:
        def test_something(check_ir):
            @ml_function
            def my_func(x: int) -> int:
                return add(x, 5)

            my_func(1)  # triggers compilation

            check_ir('''
                // CHECK: func.func @my_func
                // CHECK: arith.addi
            ''')
    """
    filecheck_bin = _find_filecheck()
    if filecheck_bin is None:
        pytest.skip("FileCheck binary not found")

    def _check(pattern: str):
        ir = backend.get_module_ir()

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.check', delete=False
        ) as f:
            f.write(pattern)
            check_file = f.name

        try:
            result = subprocess.run(
                [filecheck_bin, check_file],
                input=ir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                pytest.fail(
                    f"FileCheck failed:\n"
                    f"--- Pattern ---\n{pattern}\n"
                    f"--- Actual IR ---\n{ir}\n"
                    f"--- FileCheck stderr ---\n{result.stderr}"
                )
        finally:
            os.unlink(check_file)

    return _check


@pytest.fixture
def check_lowered_ir(backend):
    """FileCheck-based IR assertion against lowered IR after a specific pass.

    Usage:
        def test_something(check_lowered_ir):
            @ml_function
            def my_func(...) -> ...:
                ...

            my_func(...)  # triggers compilation

            check_lowered_ir('''
                // CHECK: memref.alloc
                // CHECK: memref.dealloc
            ''', after="one-shot-bufferize")
    """
    filecheck_bin = _find_filecheck()
    if filecheck_bin is None:
        pytest.skip("FileCheck binary not found")

    # Enable snapshot capture for this test
    backend.enable_snapshot_capture()

    def _check(pattern: str, after: str):
        # Ensure lowering has run (idempotent if already finalized)
        functions = backend.list_functions()
        if functions:
            backend.compiler.get_function_pointer(functions[0])

        snapshots = backend.get_lowering_snapshots()
        if not snapshots:
            pytest.fail("No lowering snapshots captured")

        # Find the snapshot after the named pass
        ir = None
        for pass_name, pass_ir in snapshots:
            if after in pass_name:
                ir = pass_ir
                break

        if ir is None:
            available = [name for name, _ in snapshots]
            pytest.fail(f"Pass '{after}' not found. Available: {available}")

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.check', delete=False
        ) as f:
            f.write(pattern)
            check_file = f.name

        try:
            result = subprocess.run(
                [filecheck_bin, check_file],
                input=ir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                pytest.fail(
                    f"FileCheck failed (after pass '{after}'):\n"
                    f"--- Pattern ---\n{pattern}\n"
                    f"--- Lowered IR ---\n{ir}\n"
                    f"--- FileCheck stderr ---\n{result.stderr}"
                )
        finally:
            os.unlink(check_file)

    return _check
