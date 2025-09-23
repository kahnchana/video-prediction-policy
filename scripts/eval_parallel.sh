#!/bin/bash

# Parallel evaluation script
# Usage: ./scripts/eval_parallel.sh [num_processes] [num_sequences]

# Set wandb environment variables
export WANDB_BASE_URL="TBA"
export WANDB_API_KEY="TBA"
export WANDB_PROJECT="calvin_eval"

# Default values
NUM_PROCESSES=${1:-4}
NUM_SEQUENCES=${2:-1000}

RUN_NAME="vpp_full_parallel"

echo "Starting parallel evaluation with:"
echo "  Processes: $NUM_PROCESSES"
echo "  Sequences: $NUM_SEQUENCES"
echo "  Run name: $RUN_NAME"
echo "  WANDB Project: $WANDB_PROJECT"

cd /export/home/repo/video-prediction-policy

python policy_evaluation/calvin_evaluate_parallel.py \
    --video_model_path ckpt/pretrained/svd-robot-calvin-ft \
    --action_model_folder ckpt/pretrained/dp-calvin \
    --clip_model_path ckpt/pretrained/clip-vit-base-patch32 \
    --calvin_abc_dir /export/xgen-video/kranasinghe/data/calvin/task_ABC_D \
    --run_name ${RUN_NAME} \
    --num_processes "$NUM_PROCESSES" \
    --num_sequences "$NUM_SEQUENCES" \
    --devices "0" 

