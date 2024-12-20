#!/bin/bash
set -x
# Setting environment variables
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Script parameters
CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((29500 + $RANDOM % 29))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Path to training script
TRAIN_SCRIPT="$(dirname "$0")/train.py"

# Check if torchrun is available
if command -v torchrun &> /dev/null; then
  echo "Using torchrun mode."
  torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nproc_per_node=$GPUS \
    $TRAIN_SCRIPT $CONFIG --launcher pytorch ${@:3}
else
  echo "Using launch mode."
  python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nproc_per_node=$GPUS \
    $TRAIN_SCRIPT $CONFIG --launcher pytorch ${@:3}
fi
