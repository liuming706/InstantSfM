#!/usr/bin/env bash
# Environment setup for running InstantSfM on this machine.
# Sourced automatically by the other scripts in this folder.
#
# Handles:
#   - conda env 'instantsfm' on PATH (override with INSTANTSFM_ENV)
#   - GCC10 shim (system GCC 9.4 cannot compile -std=c++20 CUDA extensions,
#     e.g. gsplat runtime JIT, fused-ssim, bae); shim is auto-created once
#   - CUDA headers/libs from the conda cuda-toolkit (targets/ layout)
#   - 8GB-VRAM-friendly allocator setting

ENV_DIR="${INSTANTSFM_ENV:-/root/miniconda3/envs/instantsfm}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$ENV_DIR" ]; then
    echo "[env] conda env not found: $ENV_DIR (set INSTANTSFM_ENV to override)" >&2
    return 1 2>/dev/null || exit 1
fi

# GCC10 shim: expose gcc-10/g++-10 as gcc/g++ for torch extension builds.
SHIM_DIR="$ENV_DIR/gcc10-shim"
if command -v gcc-10 >/dev/null 2>&1 && [ ! -e "$SHIM_DIR/gcc" ]; then
    mkdir -p "$SHIM_DIR"
    ln -sf "$(command -v gcc-10)" "$SHIM_DIR/gcc"
    ln -sf "$(command -v g++-10)" "$SHIM_DIR/g++"
    ln -sf "$(command -v g++-10)" "$SHIM_DIR/c++"
    ln -sf "$(command -v gcc-10)" "$SHIM_DIR/cc"
fi

export PATH="$ENV_DIR/bin:$SHIM_DIR:$PATH"
if command -v gcc-10 >/dev/null 2>&1; then
    export CC=gcc-10 CXX=g++-10
fi
export CUDA_HOME="$ENV_DIR"
export CPATH="$ENV_DIR/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$ENV_DIR/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=8.9
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH

cd "$REPO_ROOT"
