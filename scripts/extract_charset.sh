#!/bin/bash

TARGET_FONT_PATH="fonts/851Gkktt_005.ttf"
TRAIN_SPLIT_RATIO=0.8
VAL_SPLIT_RATIO=0.2
RANDOM_SEED=2025
DEVICE="cuda:1"


python extract_charset.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --device "$DEVICE"
