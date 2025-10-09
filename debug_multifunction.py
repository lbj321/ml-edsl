"""Debug script to test multiple function compilation"""

from mlir_edsl import ml_function
from mlir_edsl.backend import get_backend

print("="*60)
print("Test 1: First function (integer)")
print("="*60)

@ml_function
def add_example(x, y):
    return x + y

try:
    result1 = add_example(4, 6)
    print(f"✅ Result 1: {result1}")

    # Check MLIR after first function
    backend = get_backend()
    mlir1 = backend.get_mlir_string()
    print(f"\n📝 MLIR after first function:\n{mlir1}\n")

except Exception as e:
    print(f"❌ Error in first function: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("="*60)
print("Test 2: Second function (float)")
print("="*60)

@ml_function
def float_mul(x, y):
    return x * y

try:
    # Check MLIR before second function execution
    mlir_before = backend.get_mlir_string()
    print(f"\n📝 MLIR before second function execution:\n{mlir_before}\n")

    result2 = float_mul(2.5, 4.0)
    print(f"✅ Result 2: {result2}")

    # Check MLIR after second function
    mlir2 = backend.get_mlir_string()
    print(f"\n📝 MLIR after second function:\n{mlir2}\n")

except Exception as e:
    print(f"❌ Error in second function: {e}")
    import traceback
    traceback.print_exc()

    # Try to get MLIR even on error
    try:
        mlir_error = backend.get_mlir_string()
        print(f"\n📝 MLIR at error:\n{mlir_error}\n")
    except:
        pass
