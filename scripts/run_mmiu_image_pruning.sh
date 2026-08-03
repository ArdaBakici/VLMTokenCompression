#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"

export PROJECT_DIR
export BENCHMARK=mmiu
export PREPARE_MMIU=1
export SERVER_BACKEND=vllm-pr38888-image-pruning
export IMAGE_PRUNING_RATE="${IMAGE_PRUNING_RATE:-0.3}"
export TENSOR_PARALLEL_SIZE=1
export MODEL="${1:-${MODEL:-Qwen/Qwen3-VL-8B-Instruct}}"

exec "$SCRIPT_DIR/run_benchmark.sh" mmiu "$MODEL"
