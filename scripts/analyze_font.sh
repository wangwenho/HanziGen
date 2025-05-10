#!/bin/bash

TARGET_FONT_PATH="fonts/myfont.ttf"
REFERENCE_FONTS_DIR="fonts/jigmo/"


python analyze_font.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --reference_fonts_dir "$REFERENCE_FONTS_DIR" \
    --analyze_target_font \
    --analyze_reference_fonts
