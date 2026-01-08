# ML-EDSL Build Requirements

## System Packages (Ubuntu)

```bash
sudo apt install -y \
    build-essential \
    clang \
    cmake \
    lld \
    git \
    ninja-build \
    python3 \
    python3-pip \
    python3-venv \
    libz-dev \
    libxml2-dev
```

**Protobuf (optional - Ubuntu 24.04+ recommended):**
```bash
sudo apt install libprotobuf-dev protobuf-compiler
```
*If not installed, CMake will download and build protobuf automatically (slow first build).*

## Version Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Ubuntu | 22.04 LTS | 24.04 LTS |
| CMake | 3.20 | 3.22+ |
| LLVM/MLIR | 18.0 | 21.0+ |
| Python | 3.8 | 3.10+ |
| Clang | 14 | 16+ |
| Protobuf | 3.20 | 4.21+ (or auto-fetched 29.2.0) |

## LLVM/MLIR Build (Required)

**Location:** `~/dev/llvm-project/`

### 1. Clone LLVM
```bash
mkdir -p ~/dev && cd ~/dev
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
```

### 2. Build Configuration
```bash
mkdir build && cd build

cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON

ninja -j$(nproc)
```

**Build time:** ~30-90 minutes
**Disk space:** ~20-30 GB

### 3. Verify Build
```bash
./bin/mlir-opt --version
# Should show MLIR version
```

### Critical Paths (Used by ML-EDSL)
- **LLVM includes:** `~/dev/llvm-project/llvm/include`
- **LLVM build includes:** `~/dev/llvm-project/build/include`
- **MLIR includes:** `~/dev/llvm-project/mlir/include`
- **MLIR build includes:** `~/dev/llvm-project/build/tools/mlir/include`
- **Libraries:** `~/dev/llvm-project/build/lib/`

**Note:** ML-EDSL's `cmake/FindMLIR.cmake` expects LLVM at `~/dev/llvm-project/build/`. If you use a different location, update that file.

## Python Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

**Installed packages:**
- `pybind11>=2.10.0`
- `numpy>=1.20.0`
- `protobuf>=4.21.0`
- `pytest>=6.0.0`
- `black>=22.0.0`

## Build ML-EDSL

```bash
./build.sh
```

**First build:** 2-15 minutes (depends on if system protobuf installed)
**Incremental builds:** 15-30 seconds

## Clean Build (Rarely Needed)

```bash
./build.sh clean
./build.sh
```

Only when: CMakeLists.txt changed, LLVM version changed, or build errors.
