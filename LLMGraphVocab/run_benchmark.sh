#!/bin/bash

# Script to evaluate model performance on 7 tasks with 14 datasets
# Usage: bash evaluate_all_tasks.sh <model_path> <encoding> <splits> <samples> [batch_size]
# Example: bash evaluate_all_tasks.sh Qwen/Qwen3-14B-graphvocab-stage1-full GraphVocab 10 10 4

# Check if correct number of arguments provided
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <model_path> <encoding> <splits> <samples> [batch_size]"
    echo "  model_path: Path to the model to evaluate"
    echo "  encoding: EdgeList, GraphVocab, or Incident"
    echo "  splits: Number of splits for evaluation (e.g., 10)"
    echo "  samples: Number of samples in dataset (e.g., 10)"
    echo "  batch_size: (Optional) Batch size for inference, default is 1 (e.g., 4 for faster multi-GPU inference)"
    echo ""
    echo "Example: $0 Qwen/Qwen3-14B-graphvocab-stage1-full GraphVocab 3 100 4"
    exit 1
fi

MODEL_PATH=$1
ENCODING=$2
SPLITS=$3
SAMPLES=$4
BATCH_SIZE=${5:-1}  # Default to 1 if not provided

# Validate encoding parameter
if [ "$ENCODING" != "EdgeList" ] && [ "$ENCODING" != "GraphVocab" ] && [ "$ENCODING" != "Incident" ]; then
    echo "Error: encoding must be either 'EdgeList', 'GraphVocab', or 'Incident'"
    exit 1
fi

# Validate splits parameter (must be a positive integer)
if ! [[ "$SPLITS" =~ ^[0-9]+$ ]] || [ "$SPLITS" -le 0 ]; then
    echo "Error: splits must be a positive integer"
    exit 1
fi

# Validate samples parameter (must be a positive integer)
if ! [[ "$SAMPLES" =~ ^[0-9]+$ ]] || [ "$SAMPLES" -le 0 ]; then
    echo "Error: samples must be a positive integer"
    exit 1
fi

# Validate batch_size parameter (must be a positive integer)
if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -le 0 ]; then
    echo "Error: batch_size must be a positive integer"
    exit 1
fi

echo "=========================================="
echo "Model Evaluation Script"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Encoding: $ENCODING"
echo "Splits: $SPLITS"
echo "Samples: $SAMPLES"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Define 7 tasks with their 2 test datasets each (14 total)
# Task 1: Connectivity
echo "=========================================="
echo "Task 1/7: Connectivity"
echo "=========================================="

echo "Running Connectivity - Dataset 1 (Easy)..."
python inference.py \
    --test_data_path "s2_connectivity/${ENCODING}_Stage2_Connectivity_CoT_Nodes-11-30_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_connectivity" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""
echo "Running Connectivity - Dataset 2 (Hard)..."
python inference.py \
    --test_data_path "s2_connectivity/${ENCODING}_Stage2_Connectivity_CoT_Nodes-31-50_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_connectivity" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""

# Task 2: Degree
echo "=========================================="
echo "Task 2/7: Degree"
echo "=========================================="

echo "Running Degree - Dataset 1 (Easy)..."
python inference.py \
    --test_data_path "s2_degree/${ENCODING}_Stage2_Degree_CoT_Nodes-11-30_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_degree" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""
echo "Running Degree - Dataset 2 (Hard)..."
python inference.py \
    --test_data_path "s2_degree/${ENCODING}_Stage2_Degree_CoT_Nodes-31-50_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_degree" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""

# Task 3: Shortest Path
echo "=========================================="
echo "Task 3/7: Shortest Path"
echo "=========================================="

echo "Running Shortest Path - Dataset 1 (Easy)..."
python inference.py \
    --test_data_path "s2_shortest_path/${ENCODING}_Stage2_ShortestPath_CoT_Nodes-11-30_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_shortest_path" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""
echo "Running Shortest Path - Dataset 2 (Hard)..."
python inference.py \
    --test_data_path "s2_shortest_path/${ENCODING}_Stage2_ShortestPath_CoT_Nodes-31-50_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_shortest_path" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""

# Task 4: Isomorphism
echo "=========================================="
echo "Task 4/7: Isomorphism"
echo "=========================================="

echo "Running Isomorphism - Dataset 1 (Easy)..."
python inference.py \
    --test_data_path "s2_isomorphism/${ENCODING}_Stage2_Isomorphism_CoT_Nodes-6-10_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_isomorphism" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""
echo "Running Isomorphism - Dataset 2 (Hard)..."
python inference.py \
    --test_data_path "s2_isomorphism/${ENCODING}_Stage2_Isomorphism_CoT_Nodes-11-15_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_isomorphism" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""

# Task 5: Cycle Detection
echo "=========================================="
echo "Task 5/7: Cycle Detection"
echo "=========================================="

echo "Running Cycle Detection - Dataset 1 (Easy)..."
python inference.py \
    --test_data_path "s2_cycle_detection/${ENCODING}_Stage2_CycleDetection_CoT_Nodes-11-30_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_cycle_detection" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""
echo "Running Cycle Detection - Dataset 2 (Hard)..."
python inference.py \
    --test_data_path "s2_cycle_detection/${ENCODING}_Stage2_CycleDetection_CoT_Nodes-31-50_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_cycle_detection" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""

# Task 6: Max Clique
echo "=========================================="
echo "Task 6/7: Max Clique"
echo "=========================================="

echo "Running Max Clique - Dataset 1 (Easy)..."
python inference.py \
    --test_data_path "s2_max_clique/${ENCODING}_Stage2_MaxClique_CoT_Nodes-5-7_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_max_clique" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""
echo "Running Max Clique - Dataset 2 (Hard)..."
python inference.py \
    --test_data_path "s2_max_clique/${ENCODING}_Stage2_MaxClique_CoT_Nodes-8-10_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_max_clique" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""

# Task 7: Max Common Subgraph
echo "=========================================="
echo "Task 7/7: Max Common Subgraph"
echo "=========================================="

echo "Running Max Common Subgraph - Dataset 1 (Easy)..."
python inference.py \
    --test_data_path "s2_max_common_subgraph/${ENCODING}_Stage2_MCS_CoT_Nodes-5-7_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_max_common_subgraph" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""
echo "Running Max Common Subgraph - Dataset 2 (Hard)..."
python inference.py \
    --test_data_path "s2_max_common_subgraph/${ENCODING}_Stage2_MCS_CoT_Nodes-8-10_Samples-${SAMPLES}_Splits-${SPLITS}_Test.jsonl" \
    --model_path "$MODEL_PATH" \
    --task "s2_max_common_subgraph" \
    --num_splits $SPLITS \
    --max_new_tokens 8192 \
    --temperature 0.5 \
    --batch_size $BATCH_SIZE

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Encoding: $ENCODING"
echo "Splits: $SPLITS"
echo "Samples: $SAMPLES"
echo "Batch Size: $BATCH_SIZE"
echo "Total tasks: 7"
echo "Total datasets: 14"
echo "=========================================="
