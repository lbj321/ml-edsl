"""Base test class for MLIR tests with IR printing support"""

import os
import shutil
from mlir_edsl.backend import get_backend
from google.protobuf import text_format, json_format


# Module-level flag to track if we've cleared ir_output this session
_IR_OUTPUT_CLEARED = False
_AST_OUTPUT_CLEARED = False

# Storage for captured ASTs during tests
_captured_asts = {}  # {func_name: [protobuf_ast, ...]}


class MLIRTestBase:
    """Base class for MLIR tests with automatic IR printing/saving

    Environment Variables:
        PRINT_IR: Set to "1" to print IR after each test
        SAVE_IR: Set to "1" to save IR files
            - Raw IR files saved to ir_output/ (.mlir, _unopt.ll, _opt.ll)
            - HTML reports saved to ir_html/ (.html)
        PRINT_AST: Set to "1" to print protobuf AST after each test
        SAVE_AST: Set to "1" to save protobuf AST to files
            - Protobuf AST saved to ast_output/ (.pb.txt, .pb.json)

    Usage:
        class TestMyFeature(MLIRTestBase):
            def test_something(self):
                @ml_function
                def my_func(x: int) -> int:
                    return add(x, 5)

                result = my_func(10)
                assert result == 15
                # IR/AST automatically printed/saved if env vars set
    """

    @classmethod
    def setup_class(cls):
        """Class-level setup - runs once per test class"""
        global _IR_OUTPUT_CLEARED, _AST_OUTPUT_CLEARED

        cls.backend = get_backend()
        cls.print_ir = os.getenv("PRINT_IR", "").lower() in ("1", "true", "yes")
        cls.save_ir = os.getenv("SAVE_IR", "").lower() in ("1", "true", "yes")
        cls.print_ast = os.getenv("PRINT_AST", "").lower() in ("1", "true", "yes")
        cls.save_ast = os.getenv("SAVE_AST", "").lower() in ("1", "true", "yes")

        # Use ir_output and ir_html in project root (tests/ is sibling to mlir_edsl/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.ir_output_dir = os.path.join(project_root, "ir_output")
        cls.ir_html_dir = os.path.join(project_root, "ir_html")
        cls.ast_output_dir = os.path.join(project_root, "ast_output")

        if cls.save_ir:
            # Clear directories once per test session (not per class)
            if not _IR_OUTPUT_CLEARED:
                if os.path.exists(cls.ir_output_dir):
                    shutil.rmtree(cls.ir_output_dir)
                if os.path.exists(cls.ir_html_dir):
                    shutil.rmtree(cls.ir_html_dir)
                _IR_OUTPUT_CLEARED = True

            # Create fresh directories
            os.makedirs(cls.ir_output_dir, exist_ok=True)
            os.makedirs(cls.ir_html_dir, exist_ok=True)

        if cls.save_ast:
            # Clear AST output directory once per test session
            if not _AST_OUTPUT_CLEARED:
                if os.path.exists(cls.ast_output_dir):
                    shutil.rmtree(cls.ast_output_dir)
                _AST_OUTPUT_CLEARED = True

            # Create fresh directory
            os.makedirs(cls.ast_output_dir, exist_ok=True)

    def setup_method(self):
        """Test-level setup - runs before each test method"""
        if self.backend:
            self.backend.clear_module()

        # Install AST capture hook if AST printing/saving is enabled
        if self.print_ast or self.save_ast:
            self._install_ast_capture()

    def teardown_method(self, method):
        """Test-level teardown - runs after each test method"""
        if not self.backend:
            return

        # Get test name from the method
        test_name = method.__name__ if method else "unknown"

        try:
            if self.print_ast:
                self._print_ast(test_name)
            if self.save_ast:
                self._save_ast_files(test_name)
            if self.print_ir:
                self._print_ir()
            if self.save_ir:
                self._save_ir(test_name)
        except Exception as e:
            # Don't fail test if IR/AST printing fails
            print(f"\nWarning: Could not output IR/AST: {e}")

    def _print_ir(self):
        """Print MLIR and LLVM IR to stdout"""
        mlir = self.backend.get_mlir_string()
        llvm = self.backend.get_llvm_ir_string()

        print(f"\n{'='*60}")
        print("MLIR:")
        print('='*60)
        print(mlir)
        print(f"\n{'='*60}")
        print("LLVM IR:")
        print('='*60)
        print(llvm)

    def _save_ir(self, test_name):
        """Save MLIR and LLVM IR to files and generate HTML

        Note: C++ backend writes raw IR files to ir_output/ based on SAVE_IR env var.
        This method collects those files and generates HTML summary to ir_html/.
        """
        # Extract function name from test - look for compiled functions
        functions = self.backend.list_functions()
        if not functions:
            print(f"\nWarning: No functions compiled in test {test_name}")
            return

        # Use first compiled function name (usually only one per test)
        func_name = functions[0]

        # Read IR files written by C++ backend (in ir_output/)
        mlir_path = os.path.join(self.ir_output_dir, f"{func_name}.mlir")
        unopt_path = os.path.join(self.ir_output_dir, f"{func_name}_unopt.ll")
        opt_path = os.path.join(self.ir_output_dir, f"{func_name}_opt.ll")

        mlir_ir = self._read_file(mlir_path)
        unopt_ir = self._read_file(unopt_path)
        opt_ir = self._read_file(opt_path)

        # Generate HTML with all three IR representations (in ir_html/)
        html_path = os.path.join(self.ir_html_dir, f"{test_name}.html")
        html_content = self._generate_ir_html(test_name, func_name, mlir_ir, unopt_ir, opt_ir)
        with open(html_path, "w") as f:
            f.write(html_content)

        print(f"\nSaved HTML: {html_path}")
        print(f"Raw IR files: ir_output/{func_name}.{{mlir,_unopt.ll,_opt.ll}}")

    def _install_ast_capture(self):
        """Monkey-patch backend to capture protobuf ASTs during compilation"""
        global _captured_asts
        _captured_asts.clear()

        # Save original method
        if not hasattr(self.backend, '_original_compile_function'):
            self.backend._original_compile_function = self.backend.compile_function_from_ast

        # Wrap to capture AST
        def capture_wrapper(name, params, return_type, ast_node):
            # Serialize AST with SSA reuse
            protobuf_ast = ast_node.to_proto_with_reuse()

            # Store for later printing/saving
            if name not in _captured_asts:
                _captured_asts[name] = []
            _captured_asts[name].append(protobuf_ast)

            # Call original
            return self.backend._original_compile_function(name, params, return_type, ast_node)

        self.backend.compile_function_from_ast = capture_wrapper

    def _print_ast(self, test_name):
        """Print captured protobuf ASTs to stdout"""
        if not _captured_asts:
            print(f"\n{'='*60}")
            print(f"No ASTs captured for test: {test_name}")
            print('='*60)
            return

        for func_name, ast_list in _captured_asts.items():
            for i, ast in enumerate(ast_list):
                print(f"\n{'='*60}")
                print(f"PROTOBUF AST: {func_name} (capture {i+1})")
                print('='*60)
                print(text_format.MessageToString(ast, indent=2))

    def _save_ast_files(self, test_name):
        """Save captured protobuf ASTs to files"""
        if not _captured_asts:
            print(f"\nWarning: No ASTs captured for test {test_name}")
            return

        for func_name, ast_list in _captured_asts.items():
            for i, ast in enumerate(ast_list):
                # Save text format (human-readable)
                txt_filename = f"{func_name}_{i}.pb.txt" if len(ast_list) > 1 else f"{func_name}.pb.txt"
                txt_path = os.path.join(self.ast_output_dir, txt_filename)
                with open(txt_path, 'w') as f:
                    f.write(text_format.MessageToString(ast, indent=2))

                # Save JSON format
                json_filename = f"{func_name}_{i}.pb.json" if len(ast_list) > 1 else f"{func_name}.pb.json"
                json_path = os.path.join(self.ast_output_dir, json_filename)
                with open(json_path, 'w') as f:
                    f.write(json_format.MessageToJson(ast, indent=2))

            print(f"\nAST saved to {self.ast_output_dir}/{func_name}.*")

    def _read_file(self, path):
        """Read file contents, return empty string if file doesn't exist"""
        try:
            with open(path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return f"(File not found: {path})"

    def _generate_ir_html(self, test_name, func_name, mlir, unopt_llvm, opt_llvm):
        """Generate HTML with side-by-side IR sections"""
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

    <!-- MLIR Row (Placeholder duplicates) -->
    <div class="ir-row">
        <div class="ir-column">
            <details>
                <summary>📝 MLIR Dialect</summary>
                <pre class="mlir">{escape(mlir)}</pre>
            </details>
        </div>
        <div class="ir-column">
            <details>
                <summary>📝 MLIR Dialect (Placeholder)</summary>
                <pre class="mlir">{escape(mlir)}</pre>
            </details>
        </div>
    </div>

    <!-- LLVM IR Row (Unoptimized | Optimized) -->
    <div class="ir-row">
        <div class="ir-column">
            <details open>
                <summary>⚙️ LLVM IR (Unoptimized)</summary>
                <pre class="llvm-unopt">{escape(unopt_llvm)}</pre>
            </details>
        </div>
        <div class="ir-column">
            <details open>
                <summary>🚀 LLVM IR (Optimized O2)</summary>
                <pre class="llvm-opt">{escape(opt_llvm)}</pre>
            </details>
        </div>
    </div>
</body>
</html>"""
