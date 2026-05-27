#!/usr/bin/env bash

# Script to generate all training and test data for a specific task
# Usage: ./gen_all_data_for_task.sh <task_name> [min_test_nodes_small] [max_test_nodes_small] [min_test_nodes_large] [max_test_nodes_large] [min_train_nodes] [max_train_nodes]
# Example: ./gen_all_data_for_task.sh shortest_path
# Example: ./gen_all_data_for_task.sh shortest_path 11 30 31 50 11 50  # Custom node ranges

set -e  # Exit on error

# Check if task name is provided
if [ $# -eq 0 ]; then
    echo "Error: Task name is required"
    echo "Usage: $0 <task_name> [min_test_nodes_small] [max_test_nodes_small] [min_test_nodes_large] [max_test_nodes_large] [min_train_nodes] [max_train_nodes]"
    echo "  task_name: Task to generate data for"
    echo "  min_test_nodes_small: Minimum nodes for small test data (optional)"
    echo "  max_test_nodes_small: Maximum nodes for small test data (optional)"
    echo "  min_test_nodes_large: Minimum nodes for large test data (optional)"
    echo "  max_test_nodes_large: Maximum nodes for large test data (optional)"
    echo "  min_train_nodes: Minimum nodes for train data (optional)"
    echo "  max_train_nodes: Maximum nodes for train data (optional)"
    echo ""
    echo "Available tasks: connectivity, isomorphism, degree, shortest_path, max_common_subgraph, cycle_detection, max_clique"
    exit 1
fi

TASK_NAME=$1

# Custom node range parameters (optional)
MIN_TEST_NODES_SMALL=$2
MAX_TEST_NODES_SMALL=$3
MIN_TEST_NODES_LARGE=$4
MAX_TEST_NODES_LARGE=$5
MIN_TRAIN_NODES=$6
MAX_TRAIN_NODES=$7

# Define task-specific default parameters using case statement
case "$TASK_NAME" in
    connectivity)
        SCRIPT_NAME="gen_data/gen_data_connectivity.py"
        DEFAULT_MIN_TEST_NODES_SMALL=11
        DEFAULT_MAX_TEST_NODES_SMALL=30
        DEFAULT_MIN_TEST_NODES_LARGE=31
        DEFAULT_MAX_TEST_NODES_LARGE=50
        DEFAULT_MIN_TRAIN_NODES=11
        DEFAULT_MAX_TRAIN_NODES=50
        ;;
    isomorphism)
        SCRIPT_NAME="gen_data/gen_data_isomorphism.py"
        DEFAULT_MIN_TEST_NODES_SMALL=6
        DEFAULT_MAX_TEST_NODES_SMALL=10
        DEFAULT_MIN_TEST_NODES_LARGE=11
        DEFAULT_MAX_TEST_NODES_LARGE=15
        DEFAULT_MIN_TRAIN_NODES=6
        DEFAULT_MAX_TRAIN_NODES=12
        ;;
    degree)
        SCRIPT_NAME="gen_data/gen_data_degree.py"
        DEFAULT_MIN_TEST_NODES_SMALL=11
        DEFAULT_MAX_TEST_NODES_SMALL=30
        DEFAULT_MIN_TEST_NODES_LARGE=31
        DEFAULT_MAX_TEST_NODES_LARGE=50
        DEFAULT_MIN_TRAIN_NODES=11
        DEFAULT_MAX_TRAIN_NODES=50
        ;;
    shortest_path)
        SCRIPT_NAME="gen_data/gen_data_shortest_path.py"
        DEFAULT_MIN_TEST_NODES_SMALL=11
        DEFAULT_MAX_TEST_NODES_SMALL=30
        DEFAULT_MIN_TEST_NODES_LARGE=31
        DEFAULT_MAX_TEST_NODES_LARGE=50
        DEFAULT_MIN_TRAIN_NODES=11
        DEFAULT_MAX_TRAIN_NODES=50
        ;;
    max_common_subgraph)
        SCRIPT_NAME="gen_data/gen_data_max_common_subgraph.py"
        DEFAULT_MIN_TEST_NODES_SMALL=5
        DEFAULT_MAX_TEST_NODES_SMALL=7
        DEFAULT_MIN_TEST_NODES_LARGE=8
        DEFAULT_MAX_TEST_NODES_LARGE=10
        DEFAULT_MIN_TRAIN_NODES=5
        DEFAULT_MAX_TRAIN_NODES=10
        ;;
    cycle_detection)
        SCRIPT_NAME="gen_data/gen_data_cycle_dectection.py"
        DEFAULT_MIN_TEST_NODES_SMALL=11
        DEFAULT_MAX_TEST_NODES_SMALL=30
        DEFAULT_MIN_TEST_NODES_LARGE=31
        DEFAULT_MAX_TEST_NODES_LARGE=50
        DEFAULT_MIN_TRAIN_NODES=11
        DEFAULT_MAX_TRAIN_NODES=50
        ;;
    max_clique)
        SCRIPT_NAME="gen_data/gen_data_max_clique.py"
        DEFAULT_MIN_TEST_NODES_SMALL=5
        DEFAULT_MAX_TEST_NODES_SMALL=7
        DEFAULT_MIN_TEST_NODES_LARGE=8
        DEFAULT_MAX_TEST_NODES_LARGE=10
        DEFAULT_MIN_TRAIN_NODES=5
        DEFAULT_MAX_TRAIN_NODES=10
        ;;
    *)
        echo "Error: Unknown task '$TASK_NAME'"
        echo "Available tasks: connectivity, isomorphism, degree, shortest_path, max_common_subgraph, cycle_detection, max_clique"
        exit 1
        ;;
esac

# Use custom parameters if provided, otherwise use defaults
MIN_TEST_NODES_SMALL=${MIN_TEST_NODES_SMALL:-$DEFAULT_MIN_TEST_NODES_SMALL}
MAX_TEST_NODES_SMALL=${MAX_TEST_NODES_SMALL:-$DEFAULT_MAX_TEST_NODES_SMALL}
MIN_TEST_NODES_LARGE=${MIN_TEST_NODES_LARGE:-$DEFAULT_MIN_TEST_NODES_LARGE}
MAX_TEST_NODES_LARGE=${MAX_TEST_NODES_LARGE:-$DEFAULT_MAX_TEST_NODES_LARGE}
MIN_TRAIN_NODES=${MIN_TRAIN_NODES:-$DEFAULT_MIN_TRAIN_NODES}
MAX_TRAIN_NODES=${MAX_TRAIN_NODES:-$DEFAULT_MAX_TRAIN_NODES}

echo "=========================================="
echo "Generating data for task: $TASK_NAME"
echo "Script: $SCRIPT_NAME"
echo "Test small nodes range: $MIN_TEST_NODES_SMALL-$MAX_TEST_NODES_SMALL"
echo "Test large nodes range: $MIN_TEST_NODES_LARGE-$MAX_TEST_NODES_LARGE"
echo "Train nodes range: $MIN_TRAIN_NODES-$MAX_TRAIN_NODES"

echo "=========================================="
echo ""

# Function to run data generation
run_gen_data() {
    local split=$1
    local num_samples=$2
    local num_splits=$3
    local min_nodes=$4
    local max_nodes=$5
    local desc=$6
    
    echo ">>> Generating $desc..."
    echo "    Split: $split, Samples: $num_samples, Splits: $num_splits, Nodes: $min_nodes-$max_nodes"
    
    python $SCRIPT_NAME \
        --CoT 1 \
        --num_samples $num_samples \
        --min_nodes $min_nodes \
        --max_nodes $max_nodes \
        --split $split \
        --num_splits $num_splits
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Success"
    else
        echo "    ✗ Failed"
        exit 1
    fi
    echo ""
}

# Generate test data (easy + hard, 100 samples, 3 splits)
echo "=========================================="
echo "Step 1: Generating TEST data"
echo "=========================================="
echo ""

# Test: Easy (small nodes), 100 samples, 3 splits
run_gen_data "test" 100 3 $MIN_TEST_NODES_SMALL $MAX_TEST_NODES_SMALL "Test data (easy, 100 samples, 3 splits)"

# Test: Hard (large nodes), 100 samples, 3 splits
run_gen_data "test" 100 3 $MIN_TEST_NODES_LARGE $MAX_TEST_NODES_LARGE "Test data (hard, 100 samples, 3 splits)"

# Generate train data (10000 samples, 1 split + 2 splits)
echo "=========================================="
echo "Step 2: Generating TRAIN data"
echo "=========================================="
echo ""

# Train: 10000 samples, 1 split
run_gen_data "train" 3000 1 $MIN_TRAIN_NODES $MAX_TRAIN_NODES "Train data (3000 samples, 1 split)"

run_gen_data "train" 10000 1 $MIN_TRAIN_NODES $MAX_TRAIN_NODES "Train data (10000 samples, 1 split)"

# Train: 10000 samples, 2 splits
run_gen_data "train" 10000 2 $MIN_TRAIN_NODES $MAX_TRAIN_NODES "Train data (10000 samples, 2 splits)"

echo "=========================================="
echo "All data generation completed successfully!"
echo "=========================================="
echo ""
echo "Generated data summary:"
echo "  - 2 test datasets"
echo "  - 3 train datasets"
echo ""
