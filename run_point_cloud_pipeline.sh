#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
OUTPUT_DIR="${1:-outputs/gaussian_n100_d10_seed17}"
mkdir -p "$OUTPUT_DIR"

INPUT_NPZ="$OUTPUT_DIR/cloud_n100_d10_seed17.npz"
"$PYTHON_BIN" -m data.generate_point_cloud \
    --n 100 --d 10 --seed 17 --output "$INPUT_NPZ" --overwrite

"$PYTHON_BIN" -m examples.run_generated_point_cloud \
    --input "$INPUT_NPZ" --output "$OUTPUT_DIR/results.json" \
    | tee "$OUTPUT_DIR/run.log"