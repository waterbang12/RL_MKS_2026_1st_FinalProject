#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

: "${TASK:=Gr_shadow_train}"

python scripts/rl_games/train.py \
    --task "$TASK" \
    --num_envs 1024 \
    --headless \
    --video \
    --video_length 480 \
    --video_interval 1000
