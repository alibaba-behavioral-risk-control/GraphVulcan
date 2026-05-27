#!/usr/bin/env bash
# Stage 3: GRPO Reinforcement Learning (local run)

set -e

NUM_GPUS=8

deepspeed --num_gpus $NUM_GPUS train_s3_joint_grpo.py \
    --model_path Qwen/Qwen3-8B \
    --per_device_train_batch_size 16 \
    --num_generations 4 \
    --num_samples_per_task 3000 \
    --max_completion_length 1024 \
    --temperature 1.0 \
    --save_steps 100 \
    --beta 0.03 \
    --deepspeed ds_config/ds_config_zero3_8xH20_Qwen3-8B.json \
    --report_to tensorboard
