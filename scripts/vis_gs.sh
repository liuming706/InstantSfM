#!/usr/bin/env bash
# Visualize a trained 3DGS model in the browser (viser + nerfview viewer).
#
# Loads a checkpoint (skips training), re-runs evaluation, then keeps an
# interactive viewer running. Open the printed URL in a browser and drag to
# look around; rendering happens live on the GPU.
#
# Usage: scripts/vis_gs.sh <data_dir> [ckpt_path] [port]
#   ckpt_path: defaults to the newest checkpoint under <data_dir>/gsplat/ckpts/
#   port:      viewer port, default 8080
#
# Stop with Ctrl+C.
set -euo pipefail
source "$(dirname "$0")/env.sh"

DATA_DIR="${1:?Usage: $0 <data_dir> [ckpt_path] [port]}"
CKPT_DIR="$DATA_DIR/gsplat/ckpts"
CKPT="${2:-$(ls -t "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | head -n1 || true)}"
PORT="${3:-8080}"

if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then
    echo "[vis] no checkpoint found under $CKPT_DIR — run scripts/run_gs.sh first" >&2
    exit 1
fi

echo "[vis] serving checkpoint: $CKPT"
echo "[vis] open http://localhost:$PORT in your browser (Ctrl+C to stop)"

python instantsfm/vis/gsplat_trainer.py mcmc \
    --data_dir "$DATA_DIR" \
    --image_folder_name images \
    --data_factor 1 \
    --result_dir "$DATA_DIR/gsplat" \
    --ckpt "$CKPT" \
    --packed \
    --port "$PORT" \
    --no-disable-viewer
