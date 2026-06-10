#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

: "${TASK:=Gr_shadow_train}"

tmux new-session -d -s train "
    conda run -n isaaclab python scripts/rl_games/train.py \
        --task $TASK \
        --num_envs 1024 \
        --headless \
    
"

echo "Training started in tmux session 'train'"
echo "Attach with: tmux attach -t train"

