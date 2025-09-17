#!/bin/bash

TARGET_FONT_PATH="fonts/851Gkktt_005.ttf"
EVAL_BATCH_SIZE=2
DEVICE="cuda:1"


TARGET_FONT_NAME="$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')"

GENERATED_IMG_DIR="samples_${TARGET_FONT_NAME}/eval_outputs/gen"
GROUND_TRUTH_IMG_DIR="samples_${TARGET_FONT_NAME}/eval_outputs/gt"

python compute_metrics.py \
    --generated_img_dir "$GENERATED_IMG_DIR" \
    --ground_truth_img_dir "$GROUND_TRUTH_IMG_DIR" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE"