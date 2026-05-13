#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

: "${TASK:=Gr_shadow_play}"
: "${NUM_ENVS:=4}"
: "${VIDEO_LENGTH:=660}"

cmd=(
    python scripts/rl_games/play.py
    --task "$TASK"
    --num_envs "$NUM_ENVS"
    --headless
    --video
    --video_length "$VIDEO_LENGTH"
)

if [[ -n "${CKPT:-}" ]]; then
    cmd+=(--checkpoint "$CKPT")
fi

exec "${cmd[@]}"
