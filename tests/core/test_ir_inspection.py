"""Tests for failure-path IR capture and inspection.

Verifies that get_failure_ir() captures the partially-lowered IR when the
lowering pipeline fails, and that the failure IR is cleared between tests.
"""

import pytest
from mlir_edsl import ml_function
from mlir_edsl.ops.arithmetic import add


class TestIRInspection:
    def test_failure_ir_empty_after_success(self, backend):
        """get_failure_ir() is empty when compilation succeeds."""
        @ml_function
        def simple(x: int) -> int:
            return add(x, 1)

        simple(5)
        assert backend.get_failure_ir() == ""

    def test_failure_ir_captured_on_lowering_failure(self, backend):
        """get_failure_ir() returns the partially-lowered IR when lowering fails."""
        @ml_function
        def simple(x: int) -> int:
            return add(x, 1)

        # Inject a type-mismatched function; verifier catches it during lowering
        backend.inject_test_failure()
        with pytest.raises(RuntimeError, match="Lowering pipeline failed"):
            simple(5)

        failure_ir = backend.get_failure_ir()
        assert failure_ir, "Expected failure IR to be non-empty after lowering failure"
        assert "func.func" in failure_ir, "Failure IR should contain MLIR function ops"
        assert "__test_failure_inject__" in failure_ir

    def test_failure_ir_cleared_between_tests(self, backend):
        """clean_module() clears failure IR so it doesn't bleed between tests."""
        # The autouse clean_module fixture ran before this test, so failure IR
        # from the previous test is gone.
        assert backend.get_failure_ir() == ""

    def test_html_report_generated_on_failure(self, backend):
        """get_failure_ir() + get_module_ir() together produce useful content."""
        @ml_function
        def simple(x: int) -> int:
            return add(x, 1)

        backend.inject_test_failure()
        with pytest.raises(RuntimeError):
            simple(5)

        module_ir = backend.get_module_ir()
        failure_ir = backend.get_failure_ir()
        # Both should be valid MLIR containing the simple function
        assert "simple" in module_ir
        assert "simple" in failure_ir
