@echo off
setlocal enabledelayedexpansion

REM ==================== USER CONFIGURATIONS ====================
set "TARGET_FONT_PATH=fonts/target_font.ttf"
set "REFERENCE_FONTS_DIR=fonts/jigmo/"

REM ==================== DO NOT MODIFY BELOW ====================
python analyze_font.py ^
    --target_font_path "%TARGET_FONT_PATH%" ^
    --reference_fonts_dir "%REFERENCE_FONTS_DIR%" ^
    --analyze_target_font ^
    --analyze_reference_fonts

if %errorlevel% equ 0 (
    echo Font analysis completed successfully!
) else (
    echo Font analysis failed!
    exit /b 1
)