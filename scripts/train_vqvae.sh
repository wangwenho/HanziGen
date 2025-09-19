#!/bin/bash

TARGET_FONT_PATH="fonts/851Gkktt_005.ttf"
TRAIN_SPLIT_RATIO=0.8
VAL_SPLIT_RATIO=0.2
RANDOM_SEED=2025
BATCH_SIZE=8
LEARNING_RATE=1e-3
NUM_EPOCHS=100
DEVICE="cuda:1"
RESUME=false
USE_AMP=true


TARGET_FONT_NAME="$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')"

MODEL_SAVE_PATH="checkpoints/vqvae_${TARGET_FONT_NAME}.pth"

python train_vqvae.py \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --device "$DEVICE" \
    --resume "$RESUME" \
    --use_amp "$USE_AMP"

if [ $? -eq 0 ]; then
    echo "✅ VQ-VAE training completed successfully!"
else
    echo "❌ VQ-VAE training failed!"
    exit 1
fi