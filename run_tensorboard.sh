#!/bin/bash
set -e

tmux new-session -d -s tensorboard "
    tensorboard --logdir logs/rl_games --port 6006 --host 0.0.0.0
"

echo "TensorBoard started in tmux session 'tensorboard'"
echo "Attach with:  tmux attach -t tensorboard"
echo "Access at:    http://localhost:6006  (after SSH port forwarding)"
echo ""
echo "To forward port from your laptop:"
echo "  ssh -L 6006:localhost:6006 ubuntu@<server-ip>"
