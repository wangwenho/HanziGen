@echo off
setlocal enabledelayedexpansion

REM ==================== USER CONFIGURATIONS ====================
set "TARGET_FONT_PATH=fonts/851Gkktt_005.ttf"
set "TRAIN_SPLIT_RATIO=0.8"
set "VAL_SPLIT_RATIO=0.2"
set "SPLIT_RANDOM_SEED=2025"
set "DEVICE=cuda"

REM ==================== DO NOT MODIFY BELOW ====================
python split_dataset.py ^
    --target_font_path "%TARGET_FONT_PATH%" ^
    --split_ratios "%TRAIN_SPLIT_RATIO%" "%VAL_SPLIT_RATIO%" ^
    --split_random_seed "%SPLIT_RANDOM_SEED%" ^
    --device "%DEVICE%"

if %errorlevel% equ 0 (
    echo ✅ Charset extraction completed successfully!
) else (
    echo ❌ Charset extraction failed!
    exit /b 1
)