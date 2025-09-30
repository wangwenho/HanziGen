@echo off
setlocal enabledelayedexpansion

REM ==================== USER CONFIGURATIONS ====================
set "TARGET_FONT_PATH=fonts/851Gkktt_005.ttf"
set "TIMESTAMP=auto"
set "BLACKLEVEL=0.5"
set "TURDSIZE=2"
set "ALPHAMAX=1.0"
set "OPTTOLERANCE=0.2"

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "SAMPLE_ROOT=samples_!FILENAME!/"

if "%TIMESTAMP%"=="auto" (
    set "LATEST_INFERENCE_DIR="
    for /f "delims=" %%d in ('dir "%SAMPLE_ROOT%ldm_inference_*" /b /ad /o-d 2^>nul') do (
        if not defined LATEST_INFERENCE_DIR (
            set "LATEST_INFERENCE_DIR=%%d"
        )
    )
    if defined LATEST_INFERENCE_DIR (
        set "TIMESTAMP=!LATEST_INFERENCE_DIR:ldm_inference_=!"
        echo 🤖 Auto-detected timestamp: !TIMESTAMP!
    ) else (
        echo ❌ No inference directories found
        exit /b 1
    )
) else (
    echo 📝 Using manual timestamp: %TIMESTAMP%
)

set "INPUT_DIR=%SAMPLE_ROOT%ldm_inference_!TIMESTAMP!/infer/gen/"
set "OUTPUT_DIR=%SAMPLE_ROOT%ldm_inference_!TIMESTAMP!/svg/"

python convert_to_svg.py ^
    --input_dir "%INPUT_DIR%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --blacklevel "%BLACKLEVEL%" ^
    --turdsize "%TURDSIZE%" ^
    --alphamax "%ALPHAMAX%" ^
    --opttolerance "%OPTTOLERANCE%"

if %errorlevel% equ 0 (
    echo ✅ SVG conversion completed successfully!
) else (
    echo ❌ SVG conversion failed!
    exit /b 1
)