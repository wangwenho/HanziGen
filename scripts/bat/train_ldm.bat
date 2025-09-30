@echo off
setlocal enabledelayedexpansion

REM ==================== USER CONFIGURATIONS ====================
set "TARGET_FONT_PATH=fonts/851Gkktt_005.ttf"
set "TRAIN_SPLIT_RATIO=0.8"
set "VAL_SPLIT_RATIO=0.2"
set "SPLIT_RANDOM_SEED=2025"
set "BATCH_SIZE=16"
set "LEARNING_RATE=5e-4"
set "NUM_EPOCHS=250"
set "SAMPLE_STEPS=50"
set "IMG_SAVE_INTERVAL=5"
set "LPIPS_EVAL_INTERVAL=10"
set "EVAL_BATCH_SIZE=2"
set "DEVICE=cuda"
set "RESUME=false"
set "USE_AMP=false"

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "PRETRAINED_VQVAE_PATH=checkpoints/vqvae_!FILENAME!.pth"
set "MODEL_SAVE_PATH=checkpoints/ldm_!FILENAME!.pth"
set "TENSORBOARD_LOG_DIR=runs/LDM_!FILENAME!/"
set "SAMPLE_ROOT=samples_!FILENAME!/"

python train_ldm.py ^
    --split_ratios "%TRAIN_SPLIT_RATIO%" "%VAL_SPLIT_RATIO%" ^
    --split_random_seed "%SPLIT_RANDOM_SEED%" ^
    --batch_size "%BATCH_SIZE%" ^
    --learning_rate "%LEARNING_RATE%" ^
    --num_epochs "%NUM_EPOCHS%" ^
    --pretrained_vqvae_path "%PRETRAINED_VQVAE_PATH%" ^
    --model_save_path "%MODEL_SAVE_PATH%" ^
    --tensorboard_log_dir "%TENSORBOARD_LOG_DIR%" ^
    --sample_root "%SAMPLE_ROOT%" ^
    --sample_steps "%SAMPLE_STEPS%" ^
    --img_save_interval "%IMG_SAVE_INTERVAL%" ^
    --lpips_eval_interval "%LPIPS_EVAL_INTERVAL%" ^
    --eval_batch_size "%EVAL_BATCH_SIZE%" ^
    --device "%DEVICE%" ^
    --resume "%RESUME%" ^
    --use_amp "%USE_AMP%"

if %errorlevel% equ 0 (
    echo ✅ LDM training completed successfully!
) else (
    echo ❌ LDM training failed!
    exit /b 1
)