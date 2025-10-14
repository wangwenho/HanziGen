#!/bin/bash

# ==================== USER CONFIGURATIONS ====================
TARGET_FONT_PATH="fonts/target_font.ttf"
REFERENCE_FONTS_DIR="fonts/jigmo/"

# ==================== DO NOT MODIFY BELOW ====================
python analyze_font.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --reference_fonts_dir "$REFERENCE_FONTS_DIR" \
    --analyze_target_font \
    --analyze_reference_fonts

exit $?