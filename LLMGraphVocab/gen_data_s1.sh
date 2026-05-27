#!/usr/bin/env bash

# Script to generate Stage 1 (Structural Semantic Pretraining) data
# Usage: ./gen_data_s1.sh

set -e

echo "=========================================="
echo "Generating Stage 1: Decomp-Merge data"
echo "=========================================="
echo ""

# Generate training data with 15 relabels
echo ">>> Generating training data (num_relabel=15, max_nodes=5)..."
python gen_data/gen_data_stage1.py \
    --num_relabel 15 \
    --max_nodes 5 \
    --split train

echo ""
echo "=========================================="
echo "Stage 1 data generation completed!"
echo "=========================================="
