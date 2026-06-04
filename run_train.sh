#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

: "${TASK:=Gr_shadow_train}"

python scripts/rl_games/train.py \
  --task Gr_shadow_train \
  --num_envs 512 \
  --headless