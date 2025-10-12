#!/bin/bash

# ==================== USER CONFIGURATIONS ====================
TARGET_FONT_PATH="fonts/target_font.ttf"
TIMESTAMP="auto"
EVAL_BATCH_SIZE=2
DEVICE="cuda"

# ==================== DO NOT MODIFY BELOW ====================
TARGET_FONT_NAME="$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')"
SAMPLE_ROOT="samples_${TARGET_FONT_NAME}/"

if [ "$TIMESTAMP" = "auto" ]; then
    LATEST_LDM_DIR=$(find "${SAMPLE_ROOT}" -maxdepth 1 -name "ldm_training_*" -type d | sort -r | head -n 1)
    TIMESTAMP=$(basename "$LATEST_LDM_DIR" | sed 's/ldm_training_//')
    echo "Auto-detected timestamp: $TIMESTAMP"
else
    echo "Using manual timestamp: $TIMESTAMP"
fi

GENERATED_IMG_DIR="${SAMPLE_ROOT}ldm_training_${TIMESTAMP}/eval/gen/"
GROUND_TRUTH_IMG_DIR="${SAMPLE_ROOT}ldm_training_${TIMESTAMP}/eval/gt/"

python compute_metrics.py \
    --generated_img_dir "$GENERATED_IMG_DIR" \
    --ground_truth_img_dir "$GROUND_TRUTH_IMG_DIR" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "Metrics computation completed successfully!"
else
    echo "Metrics computation failed!"
    exit 1
fi