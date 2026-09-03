#!/usr/bin/env bash
# Train 3D Gaussian Splatting on an InstantSfM reconstruction.
#
# Uses the 8GB-VRAM-safe recipe (RTX 4060):
#   - mcmc strategy: caps Gaussians at 1M (the 'default' strategy grows
#     unbounded and OOMs on 8GB)
#   - --packed rendering + expandable_segments allocator (see env.sh)
#
# Usage: scripts/run_gs.sh <data_dir> [steps_scaler]
#   steps_scaler: fraction of the full 30000 steps (default 0.5 = 15000 steps,
#                 ~12 min on RTX 4060; use 1.0 for full quality)
#
# Requires a sparse model (run scripts/run_sfm.sh first).
# Output: checkpoints in <data_dir>/gsplat/ckpts/, renders/videos/stats alongside.
set -euo pipefail
source "$(dirname "$0")/env.sh"

DATA_DIR="${1:?Usage: $0 <data_dir> [steps_scaler]}"
STEPS_SCALER="${2:-0.5}"
RESULT_DIR="$DATA_DIR/gsplat"

if [ ! -d "$DATA_DIR/sparse/0" ]; then
    echo "[gs] no SfM model at $DATA_DIR/sparse/0 — run scripts/run_sfm.sh first" >&2
    exit 1
fi

python instantsfm/vis/gsplat_trainer.py mcmc \
    --data_dir "$DATA_DIR" \
    --image_folder_name images \
    --data_factor 1 \
    --result_dir "$RESULT_DIR" \
    --steps_scaler "$STEPS_SCALER" \
    --packed

echo "[gs] done. Checkpoints: $RESULT_DIR/ckpts/"
