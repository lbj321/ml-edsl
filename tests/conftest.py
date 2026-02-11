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


# ==================== STATE ====================

_ir_output_cleared = False


# ==================== HELPERS ====================

def _read_file(path):
    """Read file contents, return placeholder if file doesn't exist."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"(File not found: {path})"


def _generate_ir_html(test_name, func_name, mlir, unopt_llvm, opt_llvm):
    """Generate HTML with side-by-side IR sections."""
    from html import escape
    import datetime

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>IR Output: {escape(test_name)}</title>
    <style>
        body {{
            font-family: 'Consolas', 'Monaco', monospace;
            margin: 20px;
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-size: 16px;
        }}
        h1 {{
            color: #4ec9b0;
            border-bottom: 2px solid #4ec9b0;
            padding-bottom: 10px;
            font-size: 28px;
        }}
        .metadata {{
            color: #6a9955;
            font-style: italic;
            margin-bottom: 20px;
            font-size: 14px;
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
            height: 100%;
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
        .mlir {{ color: #ce9178; }}
        .llvm-unopt {{ color: #dcdcaa; }}
        .llvm-opt {{ color: #9cdcfe; }}
    </style>
</head>
<body>
    <h1>Test: {escape(test_name)}</h1>
    <div class="metadata">
        Function: {escape(func_name)}<br>
        Generated: {escape(str(datetime.datetime.now()))}
    </div>

    <!-- MLIR Row -->
    <div class="ir-row">
        <div class="ir-column">
            <details>
                <summary>MLIR Dialect</summary>
                <pre class="mlir">{escape(mlir)}</pre>
            </details>
        </div>
        <div class="ir-column">
            <details>
                <summary>MLIR Dialect (Placeholder)</summary>
                <pre class="mlir">{escape(mlir)}</pre>
            </details>
        </div>
    </div>

    <!-- LLVM IR Row (Unoptimized | Optimized) -->
    <div class="ir-row">
        <div class="ir-column">
            <details open>
                <summary>LLVM IR (Unoptimized)</summary>
                <pre class="llvm-unopt">{escape(unopt_llvm)}</pre>
            </details>
        </div>
        <div class="ir-column">
            <details open>
                <summary>LLVM IR (Optimized O2)</summary>
                <pre class="llvm-opt">{escape(opt_llvm)}</pre>
            </details>
        </div>
    </div>
</body>
</html>"""


def _save_ir_for_test(backend, test_name):
    """Save IR files and generate HTML report for a test."""
    global _ir_output_cleared

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ir_output_dir = os.path.join(project_root, "ir_output")
    ir_html_dir = os.path.join(project_root, "ir_html")

    # Clear directories once per session
    if not _ir_output_cleared:
        if os.path.exists(ir_output_dir):
            shutil.rmtree(ir_output_dir)
        if os.path.exists(ir_html_dir):
            shutil.rmtree(ir_html_dir)
        _ir_output_cleared = True

    os.makedirs(ir_output_dir, exist_ok=True)
    os.makedirs(ir_html_dir, exist_ok=True)

    functions = backend.list_functions()
    if not functions:
        return

    func_name = functions[0]

    # Read IR files written by C++ backend
    mlir_ir = _read_file(os.path.join(ir_output_dir, f"{func_name}.mlir"))
    unopt_ir = _read_file(os.path.join(ir_output_dir, "module_unopt.ll"))
    opt_ir = _read_file(os.path.join(ir_output_dir, "module_opt.ll"))

    # Generate HTML report
    html_path = os.path.join(ir_html_dir, f"{test_name}.html")
    html_content = _generate_ir_html(test_name, func_name, mlir_ir, unopt_ir, opt_ir)
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


# ==================== FIXTURES ====================

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
        if save_ir:
            try:
                _save_ir_for_test(b, request.node.name)
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
