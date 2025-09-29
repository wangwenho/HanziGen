#!/bin/bash

TARGET_FONT_PATH="fonts/851Gkktt_005.ttf"
TIMESTAMP="20250929_020227"
EVAL_BATCH_SIZE=2
DEVICE="cuda:1"

TARGET_FONT_NAME="$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')"
GENERATED_IMG_DIR="samples_${TARGET_FONT_NAME}/ldm_training_${TIMESTAMP}/eval/gen/"
GROUND_TRUTH_IMG_DIR="samples_${TARGET_FONT_NAME}/ldm_training_${TIMESTAMP}/eval/gt/"

python compute_metrics.py \
    --generated_img_dir "$GENERATED_IMG_DIR" \
    --ground_truth_img_dir "$GROUND_TRUTH_IMG_DIR" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "✅ Metric computation completed successfully!"
else
    echo "❌ Metric computation failed!"
    exit 1
fi