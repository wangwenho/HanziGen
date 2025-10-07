@echo off
setlocal enabledelayedexpansion

REM ==================== USER CONFIGURATIONS ====================
set "TARGET_FONT_PATH=fonts/target_font.ttf"
set "REFERENCE_FONTS_DIR=fonts/jigmo/"
set "CHARSET_PATH=auto"
set "BATCH_SIZE=16"
set "SAMPLE_STEPS=50"
set "IMG_WIDTH=512"
set "IMG_HEIGHT=512"
set "DEVICE=cuda"

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "PRETRAINED_LDM_PATH=checkpoints/ldm_!FILENAME!.pth"
set "SAMPLE_ROOT=samples_!FILENAME!/"

if "%CHARSET_PATH%"=="auto" (
    set "CHARSET_PATH=charsets/jf7000_coverage/!FILENAME!/missing.txt"
    echo 🤖 Auto-detected charset path: !CHARSET_PATH!
) else (
    echo 📝 Using manual charset path: %CHARSET_PATH%
)

python inference.py ^
    --target_font_path "%TARGET_FONT_PATH%" ^
    --reference_fonts_dir "%REFERENCE_FONTS_DIR%" ^
    --charset_path "%CHARSET_PATH%" ^
    --pretrained_ldm_path "%PRETRAINED_LDM_PATH%" ^
    --batch_size "%BATCH_SIZE%" ^
    --sample_root "%SAMPLE_ROOT%" ^
    --sample_steps "%SAMPLE_STEPS%" ^
    --img_size "%IMG_WIDTH%" "%IMG_HEIGHT%" ^
    --device "%DEVICE%"

if %errorlevel% equ 0 (
    echo ✅ Inference completed successfully!
) else (
    echo ❌ Inference failed!
    exit /b 1
)