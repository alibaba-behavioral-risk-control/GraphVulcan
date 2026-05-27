#!/usr/bin/env bash
# Stage 2: Multi-task Joint SFT (local run)

set -e

NUM_GPUS=4

deepspeed --num_gpus $NUM_GPUS train_s2_joint_sft.py \
    --model_path Qwen/Qwen3-8B \
    --per_device_train_batch_size 8 \
    --shuffle_seed 42 \
    --save_steps 200 \
    --report_to tensorboard \
    --deepspeed ds_config/ds_config_zero2_4xH20_Qwen3-8B.json
