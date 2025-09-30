@echo off
setlocal enabledelayedexpansion

REM ==================== USER CONFIGURATIONS ====================
set "TARGET_FONT_PATH=fonts/851Gkktt_005.ttf"
set "TIMESTAMP=auto"
set "EVAL_BATCH_SIZE=2"
set "DEVICE=cuda"

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "SAMPLE_ROOT=samples_!FILENAME!/"

if "%TIMESTAMP%"=="auto" (
    set "LATEST_LDM_DIR="
    for /f "delims=" %%d in ('dir "%SAMPLE_ROOT%ldm_training_*" /b /ad /o-d 2^>nul') do (
        if not defined LATEST_LDM_DIR (
            set "LATEST_LDM_DIR=%%d"
        )
    )
    if defined LATEST_LDM_DIR (
        set "TIMESTAMP=!LATEST_LDM_DIR:ldm_training_=!"
        echo 🤖 Auto-detected timestamp: !TIMESTAMP!
    ) else (
        echo ❌ No training directories found
        exit /b 1
    )
) else (
    echo 📝 Using manual timestamp: %TIMESTAMP%
)

set "GENERATED_IMG_DIR=%SAMPLE_ROOT%ldm_training_!TIMESTAMP!/eval/gen/"
set "GROUND_TRUTH_IMG_DIR=%SAMPLE_ROOT%ldm_training_!TIMESTAMP!/eval/gt/"

python compute_metrics.py ^
    --generated_img_dir "%GENERATED_IMG_DIR%" ^
    --ground_truth_img_dir "%GROUND_TRUTH_IMG_DIR%" ^
    --eval_batch_size "%EVAL_BATCH_SIZE%" ^
    --device "%DEVICE%"

if %errorlevel% equ 0 (
    echo ✅ Metric computation completed successfully!
) else (
    echo ❌ Metric computation failed!
    exit /b 1
)