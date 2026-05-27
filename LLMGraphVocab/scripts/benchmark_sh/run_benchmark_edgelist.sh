#!/usr/bin/env bash
# Benchmark evaluation for EdgeList encoding on 7 graph reasoning tasks

set -e

MODEL_PATH="Qwen/Qwen3-8B"
ENCODING="EdgeList"
SPLITS=3
SAMPLES=100
BATCH_SIZE=10

bash run_benchmark.sh "$MODEL_PATH" "$ENCODING" "$SPLITS" "$SAMPLES" "$BATCH_SIZE"
