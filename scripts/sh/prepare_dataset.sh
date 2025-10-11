#!/bin/bash

# ==================== USER CONFIGURATIONS ====================
TARGET_FONT_PATH="fonts/target_font.ttf"
REFERENCE_FONTS_DIR="fonts/jigmo/"
IMG_WIDTH=512
IMG_HEIGHT=512
SAMPLE_RATIO=1.0

# ==================== DO NOT MODIFY BELOW ====================
TARGET_FONT_NAME="$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')"
SOURCE_CHARSET_PATH="charsets/unihan_coverage/${TARGET_FONT_NAME}/covered.txt"

if [ -d "data" ]; then
    echo "🗑️ Removing existing data/ directory..."
    rm -rf data/
fi

python prepare_dataset.py \
    --target_font_path "$TARGET_FONT_PATH" \
    --reference_fonts_dir "$REFERENCE_FONTS_DIR" \
    --source_charset_path "$SOURCE_CHARSET_PATH" \
    --img_size "$IMG_WIDTH" "$IMG_HEIGHT" \
    --sample_ratio "$SAMPLE_RATIO"

if [ $? -eq 0 ]; then
    echo "✅ Dataset preparation completed successfully!"
else
    echo "❌ Dataset preparation failed!"
    exit 1
fi