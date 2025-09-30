@echo off
setlocal enabledelayedexpansion

REM ==================== USER CONFIGURATIONS ====================
set "TARGET_FONT_PATH=fonts/851Gkktt_005.ttf"
set "TRAIN_SPLIT_RATIO=0.8"
set "VAL_SPLIT_RATIO=0.2"
set "SPLIT_RANDOM_SEED=2025"
set "BATCH_SIZE=8"
set "LEARNING_RATE=1e-3"
set "NUM_EPOCHS=100"
set "IMG_SAVE_INTERVAL=5"
set "DEVICE=cuda"
set "RESUME=false"
set "USE_AMP=false"

REM ==================== DO NOT MODIFY BELOW ====================
for %%f in ("%TARGET_FONT_PATH%") do (
    set "FILENAME=%%~nf"
)
set "MODEL_SAVE_PATH=checkpoints/vqvae_!FILENAME!.pth"
set "TENSORBOARD_LOG_DIR=runs/VQVAE_!FILENAME!"
set "SAMPLE_ROOT=samples_!FILENAME!/"

python train_vqvae.py ^
    --split_ratios "%TRAIN_SPLIT_RATIO%" "%VAL_SPLIT_RATIO%" ^
    --split_random_seed "%SPLIT_RANDOM_SEED%" ^
    --batch_size "%BATCH_SIZE%" ^
    --learning_rate "%LEARNING_RATE%" ^
    --num_epochs "%NUM_EPOCHS%" ^
    --model_save_path "%MODEL_SAVE_PATH%" ^
    --tensorboard_log_dir "%TENSORBOARD_LOG_DIR%" ^
    --sample_root "%SAMPLE_ROOT%" ^
    --img_save_interval "%IMG_SAVE_INTERVAL%" ^
    --device "%DEVICE%" ^
    --resume "%RESUME%" ^
    --use_amp "%USE_AMP%"

if %errorlevel% equ 0 (
    echo ✅ VQ-VAE training completed successfully!
) else (
    echo ❌ VQ-VAE training failed!
    exit /b 1
)