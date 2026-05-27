#!/usr/bin/env bash

# Script to generate test/train data for all tasks
# Usage: ./gen_all_data.sh

set -e

TASKS=(
    connectivity
    isomorphism
    degree
    shortest_path
    max_common_subgraph
    cycle_detection
    max_clique
)

echo "=========================================="
echo "Generating data for ALL tasks"
echo "Tasks: ${TASKS[*]}"
echo "=========================================="
echo ""

for task in "${TASKS[@]}"; do
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "  Task: $task"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo ""
    bash gen_data/gen_data_single_task.sh "$task"
    echo ""
done

echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
