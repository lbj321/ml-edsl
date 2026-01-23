"""Symbolic execution context management"""
from contextvars import ContextVar

_symbolic_context: ContextVar[bool] = ContextVar('symbolic', default=False)


class symbolic_execution:
    """Context manager for symbolic execution mode.

    When active, @ml_function calls return CallOp AST nodes
    instead of executing. This enables nested function calls
    during AST construction.

    Usage:
        with symbolic_execution():
            result = func(*symbolic_args)  # Returns AST, not value
    """

    def __enter__(self):
        self._token = _symbolic_context.set(True)
        return self

    def __exit__(self, *args):
        _symbolic_context.reset(self._token)


def in_symbolic_context() -> bool:
    """Check if currently in symbolic execution mode."""
    return _symbolic_context.get()
