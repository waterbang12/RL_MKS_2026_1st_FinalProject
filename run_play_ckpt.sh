#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab

# List available checkpoints
echo "Available checkpoints:"
find logs/rl_games -name "*.pth" | sort
echo ""

# Use CKPT env var if set, otherwise use the latest checkpoint
if [[ -z "${CKPT:-}" ]]; then
    CKPT=$(find logs/rl_games -name "*.pth" | sort | tail -1)
    echo "Using latest checkpoint: $CKPT"
else
    echo "Using checkpoint: $CKPT"
fi

python scripts/rl_games/play.py \
    --task Gr_shadow_play \
    --num_envs 4 \
    --headless \
    --video \
    --video_length 660 \
    --checkpoint "$CKPT"
