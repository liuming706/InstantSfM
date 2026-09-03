#!/usr/bin/env bash
# Run the full InstantSfM pipeline: feature extraction + matching, then GPU SfM.
#
# Usage: scripts/run_sfm.sh <data_dir>
#   <data_dir> must contain an images/ subfolder with the input images.
#   If <data_dir>/database.db already exists, feature extraction is skipped.
#
# Output: sparse model at <data_dir>/sparse/0/
set -euo pipefail
source "$(dirname "$0")/env.sh"

DATA_DIR="${1:?Usage: $0 <data_dir>}"

if [ ! -d "$DATA_DIR/images" ]; then
    echo "[sfm] $DATA_DIR/images not found — expected a folder containing images/" >&2
    exit 1
fi

if [ -f "$DATA_DIR/database.db" ]; then
    echo "[sfm] found $DATA_DIR/database.db, skipping feature extraction"
else
    ins-feat --data_path "$DATA_DIR"
fi

ins-sfm --data_path "$DATA_DIR"
echo "[sfm] done. Sparse model: $DATA_DIR/sparse/0/"
