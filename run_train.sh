#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

# Create detached tmux session and run everything inside it
tmux new-session -d -s train "
    conda run -n isaaclab python scripts/rl_games/train.py \
        --task Gr_shadow_train \
        --num_envs 1024 \
        --headless
"

echo "Training started in tmux session 'train'"
echo "Attach with: tmux attach -t train"
