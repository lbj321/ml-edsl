#!/bin/bash

# ML-EDSL Modular Build Script
# Builds C++ MLIR backend with modular CMake structure

set -e  # Exit on error

# Default values
BUILD_TYPE="Release"
CLEAN=false
VERBOSE=false
JOBS=$(nproc 2>/dev/null || echo 4)
COMPONENT=""

# Help function
show_help() {
    cat << EOF
ML-EDSL Build Script

Usage: $0 [OPTIONS]

Options:
    clean, --clean, -c          Clean build artifacts and exit
    --debug                     Build in Debug mode (default: Release)  
    --verbose, -v              Enable verbose build output
    --jobs, -j N               Number of parallel build jobs (default: $JOBS)
    --component COMP           Build specific component (core, executor, bindings)
    --help, -h                 Show this help message

Environment Variables:
    LLVM_DIR                   Path to LLVM CMake directory
    MLIR_DIR                   Path to MLIR CMake directory

Examples:
    $0                         # Standard release build
    $0 --debug --verbose       # Debug build with verbose output
    $0 --component core        # Build only core component
    $0 clean                   # Clean all build artifacts
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        clean|--clean|-c)
            CLEAN=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --jobs|-j)
            JOBS="$2"
            shift 2
            ;;
        --component)
            COMPONENT="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Clean function
cleanup() {
    echo "🧹 Cleaning build artifacts..."
    rm -rf build/
    rm -f mlir_edsl/_mlir_backend.so
    # Clean Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    echo "✅ Cleanup complete!"
}

# If clean flag is set, clean and exit
if [ "$CLEAN" = true ]; then
    cleanup
    exit 0
fi

echo "🚀 Building ML-EDSL ($BUILD_TYPE mode)..."

# Check if LLVM_DIR and MLIR_DIR are set
if [ -z "$LLVM_DIR" ] || [ -z "$MLIR_DIR" ]; then
    echo "❌ Error: LLVM_DIR and MLIR_DIR environment variables must be set"
    echo ""
    echo "Example setup:"
    echo "  export LLVM_DIR=/path/to/llvm/lib/cmake/llvm"
    echo "  export MLIR_DIR=/path/to/llvm/lib/cmake/mlir"
    echo ""
    echo "Or pass via cmake:"
    echo "  $0 --llvm-dir=/path/to/llvm --mlir-dir=/path/to/mlir"
    exit 1
fi

# Create build directory
echo "📁 Creating build directory..."
mkdir -p build
cd build

# Configure CMake
echo "⚙️  Configuring CMake..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DLLVM_DIR="$LLVM_DIR"
    -DMLIR_DIR="$MLIR_DIR"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

if [ "$VERBOSE" = true ]; then
    CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
fi

cmake "${CMAKE_ARGS[@]}" ..

# Build
echo "🔨 Building C++ backend..."
MAKE_ARGS=()

if [ "$VERBOSE" = true ]; then
    MAKE_ARGS+=(VERBOSE=1)
fi

if [ -n "$COMPONENT" ]; then
    echo "📦 Building component: $COMPONENT"
    case $COMPONENT in
        core)
            make -j"$JOBS" "${MAKE_ARGS[@]}" mlir_edsl_core
            ;;
        executor) 
            make -j"$JOBS" "${MAKE_ARGS[@]}" mlir_edsl_executor
            ;;
        bindings)
            make -j"$JOBS" "${MAKE_ARGS[@]}" _mlir_backend
            ;;
        *)
            echo "❌ Unknown component: $COMPONENT"
            echo "Available components: core, executor, bindings"
            exit 1
            ;;
    esac
else
    make -j"$JOBS" "${MAKE_ARGS[@]}"
fi

# Install Python bindings if they were built
if [ -f "cpp/_mlir_backend.so" ]; then
    echo "📦 Installing Python bindings..."
    cp cpp/_mlir_backend.so ../mlir_edsl/
elif [ -f "_mlir_backend.so" ]; then
    echo "📦 Installing Python bindings..."
    cp _mlir_backend.so ../mlir_edsl/
fi

echo "✅ Build complete!"
echo ""
echo "🧪 To test the build:"
echo "  python3 -m pytest tests/ -v"
echo ""
echo "📚 To run examples:"  
echo "  python3 examples/simple_jit_examples.py"