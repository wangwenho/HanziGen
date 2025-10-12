#!/bin/bash

# ==================== USER CONFIGURATIONS ====================
TARGET_FONT_PATH="fonts/target_font.ttf"
TIMESTAMP="auto"
BLACKLEVEL=0.5
TURDSIZE=2
ALPHAMAX=1.0
OPTTOLERANCE=0.2

# ==================== DO NOT MODIFY BELOW ====================
TARGET_FONT_NAME="$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')"
SAMPLE_ROOT="samples_${TARGET_FONT_NAME}/"

if [ "$TIMESTAMP" = "auto" ]; then
    LATEST_INFERENCE_DIR=$(find "${SAMPLE_ROOT}" -maxdepth 1 -name "ldm_inference_*" -type d | sort -r | head -n 1)
    TIMESTAMP=$(basename "$LATEST_INFERENCE_DIR" | sed 's/ldm_inference_//')
    echo "Auto-detected timestamp: $TIMESTAMP"
else
    echo "Using manual timestamp: $TIMESTAMP"
fi

INPUT_DIR="${SAMPLE_ROOT}ldm_inference_${TIMESTAMP}/infer/gen/"
OUTPUT_DIR="${SAMPLE_ROOT}ldm_inference_${TIMESTAMP}/svg/"

python convert_to_svg.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --blacklevel "$BLACKLEVEL" \
    --turdsize "$TURDSIZE" \
    --alphamax "$ALPHAMAX" \
    --opttolerance "$OPTTOLERANCE"

if [ $? -eq 0 ]; then
    echo "SVG conversion completed successfully!"
else
    echo "SVG conversion failed!"
    exit 1
fi