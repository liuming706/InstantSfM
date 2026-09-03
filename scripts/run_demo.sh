#!/usr/bin/env bash
# Launch the gradio web demo at http://localhost:7860
# (upload images or point at a folder with images/ to reconstruct).
set -euo pipefail
source "$(dirname "$0")/env.sh"

python demo.py
