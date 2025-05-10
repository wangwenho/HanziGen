#!/bin/bash

TARGET_FONT_PATH="fonts/myfont.ttf"
BLACKLEVEL=0.5
TURDSIZE=2
ALPHAMAX=1.0
OPTTOLERANCE=0.2


TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

INPUT_DIR="samples_${TARGET_FONT_NAME}/inference/gen"
OUTPUT_DIR="svgs_${TARGET_FONT_NAME}"

python convert_to_svg.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --blacklevel "$BLACKLEVEL" \
    --turdsize "$TURDSIZE" \
    --alphamax "$ALPHAMAX" \
    --opttolerance "$OPTTOLERANCE"
