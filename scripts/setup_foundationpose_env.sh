#!/usr/bin/env bash
set -euo pipefail

# FoundationPose local setup helper for /home/originflow/project/FoundationPose.
# This script installs the local conda environment, CUDA toolkit/nvcc inside the env,
# PyTorch cu128, Python dependencies, and builds mycpp.
# Remaining manual step:
#   - FoundationPose scorer/refiner weights must be downloaded from the official Google Drive.

FP_ROOT=${FP_ROOT:-/home/originflow/project/FoundationPose}
CONDA=${CONDA:-/home/originflow/miniforge3/bin/conda}
ENV_NAME=${ENV_NAME:-foundationpose}

cd "$FP_ROOT"

if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA" env create -f environment.yml
fi

source /home/originflow/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Install CUDA compiler/toolkit inside conda env so /usr/local/cuda is not required.
"$CONDA" install -n "$ENV_NAME" -y -c nvidia -c conda-forge \
  cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-nvrtc-dev=12.8 \
  cuda-libraries-dev=12.8 cuda-profiler-api=12.8

# CUDA 12.8 rejects conda-forge gcc/g++ 14.x during PyTorch extension builds.
# Pin gcc/g++ 13.x before building nvdiffrast or mycpp.
"$CONDA" install -n "$ENV_NAME" -y -c conda-forge \
  'gcc_linux-64=13.*' 'gxx_linux-64=13.*'

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Core runtime dependencies. Open3D is installed separately because it is large and may need retry.
python -m pip install \
  scipy scikit-learn h5py joblib PyYAML ruamel.yaml opencv-python imageio trimesh \
  transformations matplotlib pandas pyrender pyOpenGL pyOpenGL_accelerate kornia \
  omegaconf psutil tqdm warp-lang scikit-image
python -m pip install open3d

# PyTorch3D: conda-forge CPU build is enough for import/check_env; GPU build may require matching conda torch/cuda stack.
"$CONDA" install -n "$ENV_NAME" -y -c conda-forge --no-deps pytorch3d=0.7.9
python -m pip install iopath

# nvdiffrast has no PyPI/conda prebuilt package; build from GitHub source.
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

# Build mycpp in an isolated Ninja build dir, then copy the .so to the path expected by check_env.py.
mkdir -p mycpp/build_ninja
cd mycpp/build_ninja
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
  -DPython3_ROOT_DIR="$CONDA_PREFIX" \
  -DPYBIND11_PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build . -j"$(nproc)"
cd "$FP_ROOT"
mkdir -p mycpp/build
cp mycpp/build_ninja/mycpp*.so mycpp/build/

python check_env.py || true
