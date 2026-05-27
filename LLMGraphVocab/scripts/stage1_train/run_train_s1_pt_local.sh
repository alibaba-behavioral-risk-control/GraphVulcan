#!/usr/bin/env bash
# Stage 1: Structural Semantic Pretraining (local run)

set -e

python train_s1.py \
    --base_model_path Qwen/Qwen3-8B \
    --dmc_dataset_path s1_decomp_merge_convert/GraphVocab_Stage1_DMC_Relabels-15_MaxNodes-5_Train.jsonl \
    --extend_graph_vocab 1 \
    --per_device_train_batch_size 16 \
    --save_steps 200
