#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

# Use CKPT env var if provided, otherwise find latest
if [[ -z "$CKPT" ]]; then
    CKPT=$(find logs/rl_games -name "*.pth" | sort | tail -1)
fi

if [[ -z "$CKPT" ]]; then
    echo "No checkpoint found. Set CKPT=/path/to/file.pth or run run_train.sh first."
    exit 1
fi

echo "Resuming from: $CKPT"

tmux new-session -d -s train "
    conda run -n isaaclab python scripts/rl_games/train.py \
        --task Gr_shadow_train \
        --num_envs 1024 \
        --headless \
        --checkpoint $CKPT
"

echo "Training started in tmux session 'train'"
echo "Attach with: tmux attach -t train"
