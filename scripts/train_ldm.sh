#!/bin/bash

TARGET_FONT_PATH="fonts/851Gkktt_005.ttf"
TRAIN_SPLIT_RATIO=0.8
VAL_SPLIT_RATIO=0.2
RANDOM_SEED=2025
BATCH_SIZE=16
LEARNING_RATE=5e-4
NUM_EPOCHS=250
SAMPLE_STEPS=50
IMG_SAVE_INTERVAL=5
LPIPS_EVAL_INTERVAL=10
EVAL_BATCH_SIZE=2
DEVICE="cuda:1"
RESUME=false
USE_AMP=false


TARGET_FONT_NAME="$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')"
PRETRAINED_VQVAE_PATH="checkpoints/vqvae_${TARGET_FONT_NAME}.pth"
MODEL_SAVE_PATH="checkpoints/ldm_${TARGET_FONT_NAME}.pth"
SAMPLE_ROOT="samples_${TARGET_FONT_NAME}/"

python train_ldm.py \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --pretrained_vqvae_path "$PRETRAINED_VQVAE_PATH" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --sample_root "$SAMPLE_ROOT" \
    --sample_steps "$SAMPLE_STEPS" \
    --img_save_interval "$IMG_SAVE_INTERVAL" \
    --lpips_eval_interval "$LPIPS_EVAL_INTERVAL" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --device "$DEVICE" \
    --resume "$RESUME" \
    --use_amp "$USE_AMP"

if [ $? -eq 0 ]; then
    echo "✅ LDM training completed successfully!"
else
    echo "❌ LDM training failed!"
    exit 1
fi