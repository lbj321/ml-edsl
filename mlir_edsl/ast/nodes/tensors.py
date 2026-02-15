"""Tensor AST nodes: TensorFromElements, TensorExtract"""

from ..base import Value
from ...types import Type, ScalarType, TensorType

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext
from .arrays import _normalize_indices, _to_scalar_node


class TensorFromElements(Value):
    """Create tensor from scalar elements.

    Maps to MLIR tensor.from_elements op.

    Compile-time type checking:
    - Validates size matches number of elements
    - Validates all elements match declared element type (strict!)

    Example:
        t = Tensor[f32, 4]([1.0, 2.0, 3.0, 4.0])
    """

    def __init__(self, elements: list, tensor_type: TensorType):
        super().__init__()
        self.elements = elements
        self.tensor_type = tensor_type

        # COMPILE-TIME TYPE CHECKING
        self._validate_size()
        self._validate_element_types()

    def _validate_size(self):
        """Ensure element count matches tensor shape."""
        if self.tensor_type.ndim == 1:
            if len(self.elements) != self.tensor_type.shape[0]:
                raise TypeError(
                    f"Tensor size mismatch: declared Tensor[{self.tensor_type.shape[0]}, ...] "
                    f"but got {len(self.elements)} elements"
                )
        elif self.tensor_type.ndim == 2:
            self.elements = self._validate_and_flatten_2d(self.elements)
        elif self.tensor_type.ndim == 3:
            self.elements = self._validate_and_flatten_3d(self.elements)

    def _validate_and_flatten_2d(self, elements):
        """Validate 2D structure and flatten to row-major order."""
        rows, cols = self.tensor_type.shape
        if not isinstance(elements, list) or len(elements) != rows:
            raise TypeError(
                f"2D tensor expects {rows} rows, got "
                f"{len(elements) if isinstance(elements, list) else 'non-list'}"
            )
        flat = []
        for i, row in enumerate(elements):
            if not isinstance(row, list) or len(row) != cols:
                raise TypeError(
                    f"Row {i}: expected {cols} elements, got "
                    f"{len(row) if isinstance(row, list) else 'non-list'}"
                )
            flat.extend(row)
        return flat

    def _validate_and_flatten_3d(self, elements):
        """Validate 3D structure and flatten to row-major order."""
        d0, d1, d2 = self.tensor_type.shape
        if not isinstance(elements, list) or len(elements) != d0:
            raise TypeError(
                f"3D tensor expects {d0} matrices, got "
                f"{len(elements) if isinstance(elements, list) else 'non-list'}"
            )
        flat = []
        for i, matrix in enumerate(elements):
            if not isinstance(matrix, list) or len(matrix) != d1:
                raise TypeError(
                    f"Matrix {i}: expected {d1} rows, got "
                    f"{len(matrix) if isinstance(matrix, list) else 'non-list'}"
                )
            for j, row in enumerate(matrix):
                if not isinstance(row, list) or len(row) != d2:
                    raise TypeError(
                        f"Matrix {i}, row {j}: expected {d2} elements, got "
                        f"{len(row) if isinstance(row, list) else 'non-list'}"
                    )
                flat.extend(row)
        return flat

    def _validate_element_types(self):
        """Ensure all elements match the declared element type (strict!)."""
        expected_type = self.tensor_type.element_type
        for i, elem in enumerate(self.elements):
            # Convert Python literals to AST nodes if needed
            elem_node = _to_scalar_node(elem)
            self.elements[i] = elem_node

            # Infer element type
            elem_type = elem_node.infer_type()

            # Strict type checking
            if elem_type != expected_type:
                raise TypeError(
                    f"Tensor element type mismatch at index {i}: "
                    f"expected {expected_type}, got {elem_type}. "
                    f"Use cast() for explicit type conversion."
                )

    def infer_type(self) -> Type:
        """TensorFromElements returns its full TensorType."""
        return self.tensor_type

    def get_children(self) -> list['Value']:
        """Return element nodes for serialization traversal."""
        return self.elements

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.tensor.from_elements.type.CopyFrom(self.tensor_type.to_proto())
        for elem in self.elements:
            pb_node.tensor.from_elements.elements.append(elem.to_proto(context))
        return pb_node


class TensorExtract(Value):
    """Extract scalar from tensor.

    Maps to MLIR tensor.extract op (value-semantic read).

    Compile-time type checking:
    - Tensor must be TensorType
    - Index count must match tensor dimensions
    - Indices must be i32
    - Result type is the tensor's element type

    Example:
        val = t[2]      # 1D extract
        val = t[1, 2]   # 2D extract
    """

    def __init__(self, tensor: Value, index):
        super().__init__()
        self.tensor = tensor
        self.indices = _normalize_indices(index)

        # COMPILE-TIME TYPE CHECKING
        self._validate_types()

    def _validate_types(self):
        """Validate tensor extract is type-safe."""
        tensor_type = self.tensor.infer_type()
        if not isinstance(tensor_type, TensorType):
            raise TypeError(
                f"Cannot index into non-tensor type. "
                f"Expected tensor, got {tensor_type}"
            )

        # Check that number of indices matches tensor dimensions
        if len(self.indices) != tensor_type.ndim:
            raise TypeError(
                f"Tensor dimension mismatch: {tensor_type.ndim}D tensor requires "
                f"{tensor_type.ndim} indices, got {len(self.indices)}. "
                f"Usage: t[i] for 1D, t[i,j] for 2D, t[i,j,k] for 3D"
            )

        # Check that all indices are i32
        for i, idx in enumerate(self.indices):
            idx_type = idx.infer_type()
            if not (isinstance(idx_type, ScalarType) and idx_type.is_integer()):
                raise TypeError(
                    f"Tensor index {i} must be i32, got {idx_type}. "
                    f"Use cast() to convert to i32."
                )

        # Store the tensor type for infer_type()
        self._tensor_type = tensor_type

    def infer_type(self) -> Type:
        """Tensor extract returns the element type (ScalarType)."""
        return self._tensor_type.element_type

    def get_children(self) -> list['Value']:
        """Return child nodes."""
        return [self.tensor] + self.indices

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.tensor.extract.tensor.CopyFrom(self.tensor.to_proto(context))
        for idx in self.indices:
            pb_node.tensor.extract.indices.append(idx.to_proto(context))
        pb_node.tensor.extract.result_type.CopyFrom(
            self._tensor_type.element_type.to_proto()
        )
        return pb_node


class TensorInsert(Value):
    """Insert scalar into tensor, returning a NEW tensor.

    Maps to MLIR tensor.insert op (value-semantic, immutable).

    Compile-time type checking:
    - Tensor must be TensorType
    - Index count must match tensor dimensions
    - Indices must be i32
    - Value type must match tensor element type exactly
    - Result is a NEW tensor with same type

    Example:
        t = Tensor[i32, 4]([1, 2, 3, 4])
        t = t.at[1].set(99)  # Returns NEW tensor [1, 99, 3, 4]
    """

    def __init__(self, tensor: Value, index, value):
        super().__init__()
        self.tensor = tensor
        self.indices = _normalize_indices(index)
        self.value = _to_scalar_node(value)

        # COMPILE-TIME TYPE CHECKING
        self._validate_types()

    def _validate_types(self):
        """Validate tensor insert is type-safe."""
        tensor_type = self.tensor.infer_type()
        if not isinstance(tensor_type, TensorType):
            raise TypeError(
                f"Cannot use .at[].set() on non-tensor type. "
                f"Expected tensor, got {tensor_type}"
            )

        # Check that number of indices matches tensor dimensions
        if len(self.indices) != tensor_type.ndim:
            raise TypeError(
                f"Tensor dimension mismatch: {tensor_type.ndim}D tensor requires "
                f"{tensor_type.ndim} indices, got {len(self.indices)}. "
                f"Usage: t.at[i].set(v) for 1D, t.at[i,j].set(v) for 2D"
            )

        # Check that all indices are i32
        for i, idx in enumerate(self.indices):
            idx_type = idx.infer_type()
            if not (isinstance(idx_type, ScalarType) and idx_type.is_integer()):
                raise TypeError(
                    f"Tensor index {i} must be i32, got {idx_type}. "
                    f"Use cast() to convert to i32."
                )

        # Check value type matches tensor element type (STRICT!)
        expected_type = tensor_type.element_type
        actual_type = self.value.infer_type()

        if actual_type != expected_type:
            raise TypeError(
                f"Cannot insert {actual_type} into "
                f"Tensor[..., {expected_type}]. "
                f"Use cast() for explicit conversion."
            )

        # Store tensor type for infer_type()
        self._tensor_type = tensor_type

    def infer_type(self) -> Type:
        """Tensor insert returns the same TensorType (new tensor, same shape)."""
        return self._tensor_type

    def get_children(self) -> list['Value']:
        """Return child nodes."""
        return [self.tensor] + self.indices + [self.value]

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.tensor.insert.tensor.CopyFrom(self.tensor.to_proto(context))
        for idx in self.indices:
            pb_node.tensor.insert.indices.append(idx.to_proto(context))
        pb_node.tensor.insert.value.CopyFrom(self.value.to_proto(context))
        pb_node.tensor.insert.result_type.CopyFrom(self._tensor_type.to_proto())
        return pb_node
