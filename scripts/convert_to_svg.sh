#!/bin/bash

TARGET_FONT_PATH="fonts/851Gkktt_005.ttf"
TIMESTAMP="20250928_103437"
BLACKLEVEL=0.5
TURDSIZE=2
ALPHAMAX=1.0
OPTTOLERANCE=0.2


TARGET_FONT_NAME="$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')"
INPUT_DIR="samples_${TARGET_FONT_NAME}/inference_${TIMESTAMP}/infer/gen/"
OUTPUT_DIR="samples_${TARGET_FONT_NAME}/svg_${TIMESTAMP}/"

python convert_to_svg.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --blacklevel "$BLACKLEVEL" \
    --turdsize "$TURDSIZE" \
    --alphamax "$ALPHAMAX" \
    --opttolerance "$OPTTOLERANCE"

if [ $? -eq 0 ]; then
    echo "✅ SVG conversion completed successfully!"
else
    echo "❌ SVG conversion failed!"
    exit 1
fi