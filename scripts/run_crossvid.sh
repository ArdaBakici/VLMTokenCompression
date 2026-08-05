#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"

export PROJECT_DIR
export BENCHMARK=crossvid
export PREPARE_CROSSVID=1
export MODEL="${1:-${MODEL:-Qwen/Qwen3-VL-8B-Instruct}}"

exec "$SCRIPT_DIR/run_benchmark.sh" crossvid "$MODEL"
