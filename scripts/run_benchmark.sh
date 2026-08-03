#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
BENCHMARK="${BENCHMARK:-${1:-}}"
MODEL="${MODEL:-${2:-Qwen/Qwen3-VL-8B-Instruct}}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
WORKERS="${WORKERS:-4}"
SYNC_ENV="${SYNC_ENV:-1}"
BOOTSTRAP_CONDA="${BOOTSTRAP_CONDA:-1}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"
SERVER_LOG_LINES="${SERVER_LOG_LINES:-200}"
STREAM_SERVER_LOGS="${STREAM_SERVER_LOGS:-1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

if [[ "$BENCHMARK" != "mmiu" && "$BENCHMARK" != "crossvid" ]]; then
    printf 'Usage: %s {mmiu|crossvid} [MODEL]\n' "$0" >&2
    printf 'Alternatively set BENCHMARK and MODEL as environment variables.\n' >&2
    exit 2
fi

cd "$PROJECT_DIR"

if [[ ! -f uv.lock ]]; then
    printf 'uv.lock not found in PROJECT_DIR=%s\n' "$PROJECT_DIR" >&2
    exit 2
fi

# Set MODULES to a space-separated site-specific module list, for example
# MODULES="anaconda cuda". No modules are assumed by default.
if [[ -n "${MODULES:-}" ]]; then
    if ! command -v module >/dev/null 2>&1; then
        printf 'MODULES was set, but the module command is unavailable.\n' >&2
        exit 2
    fi
    module purge
    read -r -a module_list <<< "$MODULES"
    for module_name in "${module_list[@]}"; do
        module load "$module_name"
    done
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-$HOME/.cache}/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/.cache}/huggingface}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${SCRATCH:-$HOME/.cache}/conda-pkgs}"
export OPENAI_API_KEY
export TOKENIZERS_PARALLELISM=false

if ! command -v conda >/dev/null 2>&1; then
    printf '%s\n' \
        'conda is not available.' \
        'Load Anaconda/Miniconda or set MODULES before running this script.' >&2
    exit 2
fi

CONDA_ENV="${CONDA_ENV:-${SCRATCH:-$PROJECT_DIR}/conda-envs/qwen3vl-bench}"
UV_BIN="$CONDA_ENV/bin/uv"
if [[ ! -x "$UV_BIN" ]]; then
    if [[ "$BOOTSTRAP_CONDA" != "1" ]]; then
        printf 'uv is missing from CONDA_ENV=%s and BOOTSTRAP_CONDA=0.\n' "$CONDA_ENV" >&2
        exit 2
    fi
    if [[ -d "$CONDA_ENV" ]]; then
        conda install --yes --prefix "$CONDA_ENV" --channel conda-forge uv
    else
        conda create --yes --prefix "$CONDA_ENV" --channel conda-forge python=3.11 uv
    fi
    SYNC_ENV=1
fi

# Install locked project packages directly into the Conda environment. --inexact
# keeps Conda's uv package instead of treating it as an extraneous dependency.
export UV_PROJECT_ENVIRONMENT="$CONDA_ENV"
if [[ "$SYNC_ENV" == "1" ]]; then
    printf 'Synchronizing the locked environment into %s\n' "$CONDA_ENV"
    "$UV_BIN" sync --frozen --inexact \
        --python "$CONDA_ENV/bin/python" \
        --extra crossvid \
        --extra serve \
        --no-dev
else
    printf 'Skipping environment synchronization because SYNC_ENV=%s\n' "$SYNC_ENV"
fi

if [[ "$BENCHMARK" == "mmiu" && "${PREPARE_MMIU:-0}" == "1" ]]; then
    MMIU_REVISION="03bf7d143d920e97a757f606b6b7baee161b019b"
    MMIU_ROOT="${MMIU_ROOT:-$PROJECT_DIR/data/MMIU}"
    MMIU_MARKER="$MMIU_ROOT/.prepared-v2-$MMIU_REVISION"
    mkdir -p "$MMIU_ROOT"

    if [[ ! -f "$MMIU_MARKER" || ! -f "$MMIU_ROOT/all.parquet" ]]; then
        printf 'Downloading MMIU revision %s into %s\n' "$MMIU_REVISION" "$MMIU_ROOT"
        "$UV_BIN" run --no-sync hf download FanqingM/MMIU-Benchmark \
            --repo-type dataset \
            --revision "$MMIU_REVISION" \
            --local-dir "$MMIU_ROOT"

        "$UV_BIN" run --no-sync python scripts/prepare_mmiu.py \
            --root "$MMIU_ROOT" \
            --marker "$MMIU_MARKER" \
            --prepare
    else
        printf 'Using prepared MMIU data in %s\n' "$MMIU_ROOT"
        "$UV_BIN" run --no-sync python scripts/prepare_mmiu.py \
            --root "$MMIU_ROOT" \
            --marker "$MMIU_MARKER"
    fi
fi

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
MODEL_TAG="${MODEL//\//_}"
RUN_DIR="${RUN_DIR:-$RESULTS_ROOT/$MODEL_TAG-$BENCHMARK}"
mkdir -p "$RUN_DIR"

SERVER_LOG="$RUN_DIR/vllm-${SLURM_JOB_ID:-local}.log"
API_BASE_URL="http://127.0.0.1:$PORT/v1"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT

printf '%s\n' '--- vLLM startup configuration ---'
printf 'Model: %s\n' "$MODEL"
printf 'Tensor parallel size: %s\n' "$TENSOR_PARALLEL_SIZE"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-<not set>}"
printf 'Maximum model length: %s\n' "$MAX_MODEL_LEN"
printf 'GPU memory utilization: %s\n' "$GPU_MEMORY_UTILIZATION"
printf 'Conda environment: %s\n' "$CONDA_ENV"
printf 'Server log: %s\n' "$SERVER_LOG"
if command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' 'Visible GPUs:'
    nvidia-smi --list-gpus || true
else
    printf '%s\n' 'Warning: nvidia-smi is not available.'
fi
printf '%s\n' '----------------------------------'

vllm_command=(
    "$UV_BIN" run --no-sync vllm serve "$MODEL"
    --host 127.0.0.1
    --port "$PORT"
    --dtype bfloat16
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --limit-mm-per-prompt '{"image":128}'
)

printf 'Starting vLLM and waiting up to %s seconds for readiness.\n' "$SERVER_START_TIMEOUT"
if [[ "$STREAM_SERVER_LOGS" == "1" ]]; then
    "${vllm_command[@]}" > >(tee "$SERVER_LOG") 2>&1 &
else
    "${vllm_command[@]}" >"$SERVER_LOG" 2>&1 &
fi
SERVER_PID=$!

if ! "$UV_BIN" run --no-sync python - \
    "$API_BASE_URL/models" "$SERVER_PID" "$SERVER_START_TIMEOUT" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

url = sys.argv[1]
server_pid = int(sys.argv[2])
timeout = int(sys.argv[3])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                print(f"vLLM is ready at {url}", flush=True)
                raise SystemExit(0)
    except (OSError, urllib.error.URLError):
        pass
    try:
        os.kill(server_pid, 0)
    except OSError:
        raise SystemExit("vLLM exited before becoming ready")
    stat_path = Path(f"/proc/{server_pid}/stat")
    if stat_path.exists() and stat_path.read_text().split()[2] == "Z":
        raise SystemExit("vLLM exited before becoming ready")
    time.sleep(5)
raise SystemExit(f"Timed out waiting {timeout} seconds for vLLM")
PY
then
    printf '\n%s\n' 'ERROR: vLLM failed to become ready.' >&2
    if wait "$SERVER_PID"; then
        server_status=0
    else
        server_status=$?
    fi
    SERVER_PID=""
    printf 'vLLM exit status: %s\n' "$server_status" >&2
    printf 'Full server log: %s\n' "$SERVER_LOG" >&2
    if [[ -s "$SERVER_LOG" ]]; then
        printf '%s\n' "--- Last $SERVER_LOG_LINES vLLM log lines ---" >&2
        tail -n "$SERVER_LOG_LINES" "$SERVER_LOG" >&2
        printf '%s\n' '--- End vLLM log ---' >&2
    else
        printf '%s\n' 'The vLLM log is empty; check Conda, CUDA, and executable availability.' >&2
    fi
    exit 1
fi

case "$BENCHMARK" in
    mmiu)
        MMIU_ROOT="${MMIU_ROOT:-$PROJECT_DIR/data/MMIU}"
        mmiu_arguments=(
            --model "$MODEL"
            --base-url "$API_BASE_URL"
            --api-key "$OPENAI_API_KEY"
            --media-root "$MMIU_ROOT"
            --dataset-path "$MMIU_ROOT/all.parquet"
            --output "$RUN_DIR/results.jsonl"
            --workers "$WORKERS"
        )
        if [[ -n "${MMIU_LIMIT:-}" ]]; then
            mmiu_arguments+=(--limit "$MMIU_LIMIT")
        fi
        "$UV_BIN" run --no-sync mmiu-eval run "${mmiu_arguments[@]}"
        ;;
    crossvid)
        CROSSVID_ROOT="${CROSSVID_ROOT:-$PROJECT_DIR/data/CrossVid}"
        CROSSVID_TASK="${CROSSVID_TASK:-all}"
        "$UV_BIN" run --no-sync crossvid-eval run \
            --task "$CROSSVID_TASK" \
            --model "$MODEL" \
            --base-url "$API_BASE_URL" \
            --api-key "$OPENAI_API_KEY" \
            --qa-dir "$CROSSVID_ROOT/QA" \
            --video-root "$CROSSVID_ROOT/videos" \
            --uav-root "$CROSSVID_ROOT/uav" \
            --results-dir "$RUN_DIR" \
            --workers "$WORKERS" \
            --frames "${FRAMES:-128}" \
            --length "${FRAME_LENGTH:-360}"

        if [[ "$CROSSVID_TASK" == "all" && -n "${JUDGE_MODEL:-}" ]]; then
            "$UV_BIN" run --no-sync crossvid-eval judge \
                --model "$JUDGE_MODEL" \
                --base-url "${JUDGE_BASE_URL:-$API_BASE_URL}" \
                --api-key "${JUDGE_API_KEY:-$OPENAI_API_KEY}" \
                --qa-dir "$CROSSVID_ROOT/QA" \
                --results-dir "$RUN_DIR" \
                --workers "${JUDGE_WORKERS:-$WORKERS}"
            "$UV_BIN" run --no-sync crossvid-score score \
                --qa-dir "$CROSSVID_ROOT/QA" \
                --results-dir "$RUN_DIR" \
                --json-output "$RUN_DIR/summary.json"
        elif [[ "$CROSSVID_TASK" == "all" ]]; then
            printf '%s\n' \
                'CrossVid inference completed without CCQA judging.' \
                'Set JUDGE_MODEL and optionally JUDGE_BASE_URL, then run the judge and scorer.'
        fi
        ;;
esac

printf 'Benchmark output: %s\n' "$RUN_DIR"
