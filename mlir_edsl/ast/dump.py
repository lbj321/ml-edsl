"""AST dump utility for inspecting symbolic expression trees.

Produces indented tree output with SSA labels for shared values.
Uses SerializationContext for use-counting to stay in sync with
protobuf serialization.
"""

from .serialization import SerializationContext, OP_NAMES, PREDICATE_NAMES


def dump(value) -> str:
    """Dump AST as indented tree with SSA for shared values.

    Args:
        value: Root Value node to dump.

    Returns:
        Multi-line string with indented tree representation.
    """
    ctx = SerializationContext()
    ctx.count_uses(value)

    lines = []
    _dump_node(value, lines, ctx, prefix="", child_prefix="")
    return "\n".join(lines)


def _label(value) -> str:
    """Compute the display label for a node."""
    from .nodes.scalars import Constant, IndexConstant, BinaryOp, CompareOp, CastOp
    from .nodes.arrays import ArrayLiteral, ArrayAccess, ArrayStore, ArrayBinaryOp
    from .nodes.control_flow import IfOp, ForLoopOp
    from .nodes.functions import Parameter, CallOp
    from .nodes.tensors import TensorFromElements, TensorExtract, TensorInsert

    t = value.infer_type()

    # IndexConstant before Constant (subclass first)
    if isinstance(value, IndexConstant):
        return f"iconst {value.value} : index"
    if isinstance(value, Constant):
        return f"const {value.value} : {t}"
    if isinstance(value, Parameter):
        return f'param "{value.name}" : {t}'
    if isinstance(value, BinaryOp):
        return f"{OP_NAMES.get(value.op, value.op)} : {t}"
    if isinstance(value, CompareOp):
        return f"cmp.{PREDICATE_NAMES.get(value.predicate, value.predicate)} : {t}"
    if isinstance(value, CastOp):
        return f"cast {value.value.infer_type()} -> {value.target_type}"
    if isinstance(value, IfOp):
        return f"if : {t}"
    if isinstance(value, ForLoopOp):
        return f"for({OP_NAMES.get(value.operation, value.operation)}) : {t}"
    if isinstance(value, ArrayLiteral):
        return f"array : {t}"
    if isinstance(value, ArrayAccess):
        return f"load : {t}"
    if isinstance(value, ArrayStore):
        return f"store : {t}"
    if isinstance(value, ArrayBinaryOp):
        return f"array.{OP_NAMES.get(value.op, value.op)} : {t}"
    if isinstance(value, TensorFromElements):
        return f"tensor.from_elements : {t}"
    if isinstance(value, TensorExtract):
        return f"tensor.extract : {t}"
    if isinstance(value, TensorInsert):
        return f"tensor.insert : {t}"
    if isinstance(value, CallOp):
        return f"call @{value.func_name} : {t}"
    return f"{value.__class__.__name__} : {t}"


def _dump_node(value, lines, ctx, prefix, child_prefix):
    """Recursively dump a node and its children."""
    # Already printed — emit a reference
    if ctx.is_serialized(value):
        lines.append(f"{prefix}%{value.id}")
        return

    label = _label(value)
    if ctx.is_reused(value):
        label = f"%{value.id} = {label}"
        ctx.mark_serialized(value)

    children = value.get_children()
    lines.append(f"{prefix}{label}")

    for i, child in enumerate(children):
        is_last = (i == len(children) - 1)
        connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "
        extension = "   " if is_last else "\u2502  "
        _dump_node(child, lines, ctx,
                   child_prefix + connector,
                   child_prefix + extension)
