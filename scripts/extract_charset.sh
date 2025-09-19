#!/bin/bash

TARGET_FONT_PATH="fonts/851.ttf"
TRAIN_SPLIT_RATIO=0.8
VAL_SPLIT_RATIO=0.2
RANDOM_SEED=2025
DEVICE="cuda:1"


python extract_charset.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "✅ Charset extraction completed successfully!"
else
    echo "❌ Charset extraction failed!"
    exit 1
fi
