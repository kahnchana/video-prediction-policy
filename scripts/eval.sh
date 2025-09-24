#!/bin/bash

# conda activate calvin

export WANDB_PROJECT="calvin_eval"

export CUDA_VISIBLE_DEVICES=0

run_name=vpp_test

python policy_evaluation/calvin_evaluate.py \
    --video_model_path ckpt/pretrained/svd-robot-calvin-ft \
    --action_model_folder ckpt/pretrained/dp-calvin \
    --clip_model_path ckpt/pretrained/clip-vit-base-patch32 \
    --calvin_abc_dir /export/xgen-video/kranasinghe/data/calvin/task_ABC_D \
    --run_name ${run_name} \
    |& tee ckpt/logs//${run_name}_$(date +%Y%m%d_%H%M%S).log
