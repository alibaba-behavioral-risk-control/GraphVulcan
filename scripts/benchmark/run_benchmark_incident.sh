#!/usr/bin/env bash
# Benchmark evaluation for Incident encoding on 7 graph reasoning tasks

set -e

MODEL_PATH="Qwen/Qwen3-8B"
ENCODING="Incident"
SPLITS=3
SAMPLES=100
BATCH_SIZE=10
DEVICE="cuda:0"

bash run_benchmark.sh "$MODEL_PATH" "$ENCODING" "$SPLITS" "$SAMPLES" "$BATCH_SIZE" "$DEVICE"
