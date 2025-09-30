@echo off
setlocal enabledelayedexpansion

REM ==================== USER CONFIGURATIONS ====================
set "TARGET_FONT_PATH=fonts/851Gkktt_005.ttf"
set "REFERENCE_FONTS_DIR=fonts/jigmo/"
set "IMG_WIDTH=512"
set "IMG_HEIGHT=512"
set "SAMPLE_RATIO=1.0"

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "SOURCE_CHARSET_PATH=charsets/unihan_coverage/!FILENAME!/covered.txt"

if exist "data" (
    echo 🗑️  Removing existing data/ directory...
    rmdir /s /q "data"
)

python prepare_dataset.py ^
    --target_font_path "%TARGET_FONT_PATH%" ^
    --reference_fonts_dir "%REFERENCE_FONTS_DIR%" ^
    --source_charset_path "%SOURCE_CHARSET_PATH%" ^
    --img_size "%IMG_WIDTH%" "%IMG_HEIGHT%" ^
    --sample_ratio "%SAMPLE_RATIO%"

if %errorlevel% equ 0 (
    echo ✅ Dataset preparation completed successfully!
) else (
    echo ❌ Dataset preparation failed!
    exit /b 1
)