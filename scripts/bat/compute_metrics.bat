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

if "!TIMESTAMP!" == "auto" (
    set "LATEST_LDM_DIR="
    set "LATEST_TIME=0"
    
    for /d %%d in ("!SAMPLE_ROOT!ldm_training_*") do (
        set "DIR_NAME=%%~nd"
        set "CURRENT_TIMESTAMP=!DIR_NAME:ldm_training_=!"
        if "!CURRENT_TIMESTAMP!" gtr "!LATEST_TIME!" (
            set "LATEST_TIME=!CURRENT_TIMESTAMP!"
            set "LATEST_LDM_DIR=%%d"
        )
    )
    
    set "TIMESTAMP=!LATEST_TIME!"
    echo 🤖 Auto-detected timestamp: !TIMESTAMP!
) else (
    echo 📝 Using manual timestamp: !TIMESTAMP!
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