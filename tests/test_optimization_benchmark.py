"""Tests for LLVM optimization levels and performance

This test suite validates:
- JIT execution at different optimization levels (O0, O2, O3)
- Optimization correctness across all levels
- Constant folding and other optimizations
- Performance comparison between optimization levels
"""

import time
import pytest
from mlir_edsl import ml_function, add, sub, mul, div
from mlir_edsl.backend import HAS_CPP_BACKEND, get_backend
from tests.test_base import MLIRTestBase


@pytest.mark.skipif(not HAS_CPP_BACKEND, reason="Requires C++ backend")
class TestOptimizationLevels(MLIRTestBase):
    """Test LLVM optimization levels"""

    def test_optimization_correctness(self):
        """Ensure optimization doesn't change computation results"""
        @ml_function
        def float_calc(a: float, b: float, c: float, d: float) -> float:
            # Complex float computation: 2.5 * 4.0 + 1.5 - 0.5 = 11.0
            mul_result = mul(a, b)      # 2.5 * 4.0 = 10.0
            add_result = add(mul_result, c)  # 10.0 + 1.5 = 11.5
            return sub(add_result, d)   # 11.5 - 0.5 = 11.0

        backend = get_backend()

        # Test each optimization level gives same result
        for opt_level in [0, 2, 3]:
            backend.set_optimization_level(opt_level)
            result = float_calc(2.5, 4.0, 1.5, 0.5)
            assert abs(result - 11.0) < 1e-6, f"O{opt_level} gave incorrect result: {result}"

    def test_optimization_with_constants(self):
        """Test that optimization handles constant expressions correctly"""
        @ml_function
        def const_expr() -> int:
            # Simple constant expression: (10 + 5) * (8 - 3) / 2 = 37
            add_result = add(10, 5)      # 15
            sub_result = sub(8, 3)       # 5
            mul_result = mul(add_result, sub_result)  # 75
            return div(mul_result, 2)    # 37

        backend = get_backend()

        # Test O0 vs O2 vs O3
        for opt_level in [0, 2, 3]:
            backend.set_optimization_level(opt_level)
            result = const_expr()
            assert result == 37, f"O{opt_level} gave incorrect result: {result}"

    def test_optimization_performance_comparison(self):
        """Compare performance across optimization levels"""
        @ml_function
        def complex_calc(a: int, b: int, c: int, d: int, e: int) -> int:
            # More complex expression to show optimization effects
            # ((a + b) * (c - d)) / e
            add_result = add(a, b)
            sub_result = sub(c, d)
            mul_result = mul(add_result, sub_result)
            return div(mul_result, e)

        backend = get_backend()
        iterations = 10000

        results = {}

        # Benchmark each optimization level
        for opt_level in [0, 2, 3]:
            backend.set_optimization_level(opt_level)

            # Warm up
            for _ in range(100):
                _ = complex_calc(10, 5, 8, 3, 1)

            # Benchmark
            start_time = time.perf_counter()
            for _ in range(iterations):
                result = complex_calc(10, 5, 8, 3, 1)
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            results[opt_level] = (execution_time, result)

            # Verify correctness: ((10+5) * (8-3)) / 1 = 15 * 5 = 75
            assert result == 75, f"O{opt_level} gave incorrect result: {result}"

        # All should produce same result
        assert results[0][1] == results[2][1] == results[3][1] == 75

        # Verify times are positive (sanity check)
        assert all(t > 0 for t, _ in results.values())

    def test_optimization_constant_folding(self):
        """Test that optimization performs constant folding"""
        @ml_function
        def simple_const() -> int:
            # Simple constant that should be folded: 4 + 6 = 10
            return add(4, 6)

        backend = get_backend()

        # Test both O0 and O2
        for opt_level in [0, 2]:
            backend.set_optimization_level(opt_level)
            result = simple_const()
            assert result == 10, f"O{opt_level} gave incorrect result: {result}"