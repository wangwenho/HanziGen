#!/bin/bash

TARGET_FONT_PATH="fonts/851Gkktt_005.ttf"
REFERENCE_FONTS_DIR="fonts/jigmo/"
BATCH_SIZE=16
SAMPLE_STEPS=50
IMG_WIDTH=512
IMG_HEIGHT=512
DEVICE="cuda:1"


TARGET_FONT_NAME="$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')"

CHARSET_PATH="charsets/jf7000_coverage/${TARGET_FONT_NAME}/missing.txt"
PRETRAINED_LDM_PATH="checkpoints/ldm_${TARGET_FONT_NAME}.pth"
SAMPLE_ROOT="samples_${TARGET_FONT_NAME}/"


python inference.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --reference_fonts_dir "$REFERENCE_FONTS_DIR" \
    --charset_path "$CHARSET_PATH" \
    --pretrained_ldm_path "$PRETRAINED_LDM_PATH" \
    --batch_size "$BATCH_SIZE" \
    --sample_root "$SAMPLE_ROOT" \
    --sample_steps "$SAMPLE_STEPS" \
    --img_size "$IMG_WIDTH" "$IMG_HEIGHT" \
    --device "$DEVICE"