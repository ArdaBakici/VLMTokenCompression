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
SERVER_BACKEND="${SERVER_BACKEND:-vllm}"
IMAGE_PRUNING_RATE="${IMAGE_PRUNING_RATE:-0.3}"
VIT_ATTENTION_SCORE_LAYER_INDEX="${VIT_ATTENTION_SCORE_LAYER_INDEX:--2}"
IMAGE_PRUNING_ENCODER_PATCH="${IMAGE_PRUNING_ENCODER_PATCH:-1}"
CROSSVID_REPOSITORY="https://github.com/chuntianli666/CrossVid.git"
CROSSVID_COMMIT="b53ada63551f9ac4a726b381b627d17ece066281"
CROSSVID_VENDOR_ROOT="${CROSSVID_VENDOR_ROOT:-$PROJECT_DIR/vendor/CrossVid}"
IMAGE_PRUNING_VLLM_REPOSITORY="https://github.com/shhn1/vllm.git"
IMAGE_PRUNING_VLLM_COMMIT="d093d3037350eb7c9de1d149f9311432de2e0adb"
IMAGE_PRUNING_VLLM_BASE_COMMIT="4eefbf9609e5ddb996e3ac37e192e92466ec35cc"

if [[ "$BENCHMARK" != "mmiu" && "$BENCHMARK" != "crossvid" ]]; then
    printf 'Usage: %s {mmiu|crossvid} [MODEL]\n' "$0" >&2
    printf 'Alternatively set BENCHMARK and MODEL as environment variables.\n' >&2
    exit 2
fi

case "$SERVER_BACKEND" in
    vllm)
        ;;
    vllm-pr38888-image-pruning)
        if [[ "$BENCHMARK" != "mmiu" ]]; then
            printf 'The experimental image-pruning profile currently supports only MMIU.\n' >&2
            exit 2
        fi
        # Both checkpoints share the 27-layer Qwen3-VL vision tower that the PR
        # scores, so VIT_ATTENTION_SCORE_LAYER_INDEX means the same thing for
        # each. Other checkpoints are unvalidated: a different vision depth
        # silently changes which layer the attention scores come from.
        image_pruning_models=(
            "Qwen/Qwen3-VL-8B-Instruct"
            "Qwen/Qwen3-VL-30B-A3B-Instruct"
        )
        if [[ ! " ${image_pruning_models[*]} " == *" $MODEL "* ]]; then
            printf 'The experimental image-pruning profile supports only:\n' >&2
            printf '  %s\n' "${image_pruning_models[@]}" >&2
            exit 2
        fi
        if [[ "$TENSOR_PARALLEL_SIZE" != "1" ]]; then
            printf '%s\n' \
                'The experimental image-pruning backend is restricted to TENSOR_PARALLEL_SIZE=1.' \
                'Tensor-parallel token selection is not validated by the upstream PR.' >&2
            exit 2
        fi
        ;;
    *)
        printf 'Unsupported SERVER_BACKEND=%s\n' "$SERVER_BACKEND" >&2
        exit 2
        ;;
esac

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

# Directory holding the per-backend Conda prefixes. Set CONDA_ENV_DIR to move
# every environment at once, for example to "$HOME/.conda/envs" so that conda
# lists them as named environments. Set CONDA_ENV to place one environment
# explicitly, keeping in mind that the two server backends must not share a
# prefix.
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${SCRATCH:-$PROJECT_DIR}/conda-envs}"
if [[ "$SERVER_BACKEND" == "vllm-pr38888-image-pruning" ]]; then
    default_conda_env="$CONDA_ENV_DIR/qwen3vl-image-pruning-d093d3037"
else
    default_conda_env="$CONDA_ENV_DIR/qwen3vl-bench"
fi
CONDA_ENV="${CONDA_ENV:-$default_conda_env}"
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
    if [[ "$SERVER_BACKEND" == "vllm" ]]; then
        "$UV_BIN" sync --frozen --inexact \
            --python "$CONDA_ENV/bin/python" \
            --extra crossvid \
            --extra serve \
            --no-dev
    else
        VLLM_USE_PRECOMPILED=1 \
        VLLM_PRECOMPILED_WHEEL_COMMIT="$IMAGE_PRUNING_VLLM_BASE_COMMIT" \
            "$UV_BIN" sync --frozen --inexact \
                --python "$CONDA_ENV/bin/python" \
                --extra crossvid \
                --extra image-pruning \
                --no-dev
    fi
else
    printf 'Skipping environment synchronization because SYNC_ENV=%s\n' "$SYNC_ENV"
fi

if [[ "$SERVER_BACKEND" == "vllm-pr38888-image-pruning" ]]; then
    VLLM_VERSION="$("$CONDA_ENV/bin/python" - \
        "$IMAGE_PRUNING_RATE" "$VIT_ATTENTION_SCORE_LAYER_INDEX" <<'PY'
from importlib.metadata import version
import sys

from vllm.config.multimodal import MultiModalConfig

vllm_version = version("vllm")
expected_version = "0.1.dev15469+gd093d3037.precompiled"
if vllm_version != expected_version:
    raise SystemExit(f"The active vLLM is not the pinned PR build: {vllm_version}")
if "image_pruning_rate" not in MultiModalConfig.__pydantic_fields__:
    raise SystemExit("The active vLLM does not expose image_pruning_rate")

try:
    pruning_rate = float(sys.argv[1])
except ValueError as exc:
    raise SystemExit("IMAGE_PRUNING_RATE must be a number") from exc
if not 0.0 < pruning_rate < 1.0:
    raise SystemExit("IMAGE_PRUNING_RATE must be greater than 0 and less than 1")

try:
    layer_index = int(sys.argv[2])
except ValueError as exc:
    raise SystemExit("VIT_ATTENTION_SCORE_LAYER_INDEX must be an integer") from exc
if not -27 <= layer_index <= -1:
    raise SystemExit(
        "VIT_ATTENTION_SCORE_LAYER_INDEX must select one of the 27 vision layers "
        "using an index from -27 through -1"
    )
print(vllm_version)
PY
    )"
    # The pinned PR build needs two local corrections before it serves MMIU:
    # the vision tower must encode pruned images one at a time to stay inside
    # the profiled memory budget, and the MoE checkpoint must assign the image
    # pruning attribute that its inherited image path reads. See
    # scripts/patch_image_pruning.py.
    if [[ "$IMAGE_PRUNING_ENCODER_PATCH" == "1" ]]; then
        ENCODER_PATCH="$("$CONDA_ENV/bin/python" \
            scripts/patch_image_pruning.py --print-id)"
        "$CONDA_ENV/bin/python" scripts/patch_image_pruning.py
    else
        ENCODER_PATCH="none"
        printf '%s\n' \
            'WARNING: IMAGE_PRUNING_ENCODER_PATCH=0 leaves the local vLLM patches' \
            'unapplied. Multi-image prompts can exhaust device memory in the' \
            'vision tower, and MoE checkpoints fail on their first image.' >&2
    fi

    BACKEND_SIGNATURE="$SERVER_BACKEND@$VLLM_VERSION;source=$IMAGE_PRUNING_VLLM_COMMIT;native=$IMAGE_PRUNING_VLLM_BASE_COMMIT;rate=$IMAGE_PRUNING_RATE;layer=$VIT_ATTENTION_SCORE_LAYER_INDEX;chunked-prefill=false;encoder-patch=$ENCODER_PATCH"
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

if [[ "$BENCHMARK" == "crossvid" && "${PREPARE_CROSSVID:-0}" == "1" ]]; then
    CROSSVID_DATASET_REVISION="4cc98eee034e6f3950c19803485402661f54c1f8"
    CROSSVID_ROOT="${CROSSVID_ROOT:-$PROJECT_DIR/data/CrossVid}"
    CROSSVID_MARKER="$CROSSVID_ROOT/.prepared-$CROSSVID_DATASET_REVISION"
    mkdir -p "$CROSSVID_ROOT"

    # crossvid_eval.py imports the official media preprocessors, so the pinned
    # upstream checkout is a hard requirement rather than a provenance copy.
    if [[ ! -d "$CROSSVID_VENDOR_ROOT/.git" ]]; then
        if [[ -e "$CROSSVID_VENDOR_ROOT" ]]; then
            printf '%s\n' \
                "CROSSVID_VENDOR_ROOT=$CROSSVID_VENDOR_ROOT exists but is not a" \
                'git checkout. Remove it or point CROSSVID_VENDOR_ROOT elsewhere.' >&2
            exit 2
        fi
        printf 'Cloning CrossVid into %s\n' "$CROSSVID_VENDOR_ROOT"
        git clone "$CROSSVID_REPOSITORY" "$CROSSVID_VENDOR_ROOT"
    fi
    if [[ "$(git -C "$CROSSVID_VENDOR_ROOT" rev-parse HEAD)" != "$CROSSVID_COMMIT" ]]; then
        printf 'Checking out CrossVid %s\n' "$CROSSVID_COMMIT"
        git -C "$CROSSVID_VENDOR_ROOT" fetch --quiet origin "$CROSSVID_COMMIT" || \
            git -C "$CROSSVID_VENDOR_ROOT" fetch --quiet origin
        git -C "$CROSSVID_VENDOR_ROOT" checkout --quiet "$CROSSVID_COMMIT"
    fi

    if [[ ! -f "$CROSSVID_MARKER" || ! -d "$CROSSVID_ROOT/QA" ]]; then
        printf '%s\n' \
            "Downloading CrossVid $CROSSVID_DATASET_REVISION into $CROSSVID_ROOT." \
            'The pinned release is about 312 GB. The download resumes if interrupted.'
        "$UV_BIN" run --no-sync hf download Chuntianli/CrossVid \
            --repo-type dataset \
            --revision "$CROSSVID_DATASET_REVISION" \
            --local-dir "$CROSSVID_ROOT"
    else
        printf 'Using prepared CrossVid data in %s\n' "$CROSSVID_ROOT"
    fi

    prepare_crossvid_arguments=(
        --root "$CROSSVID_ROOT"
        --marker "$CROSSVID_MARKER"
    )
    if [[ "${CROSSVID_ALLOW_MISSING_BEHAVIOR:-0}" == "1" ]]; then
        prepare_crossvid_arguments+=(--allow-missing-behavior)
    fi
    "$UV_BIN" run --no-sync python scripts/prepare_crossvid.py \
        "${prepare_crossvid_arguments[@]}"
fi

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
MODEL_TAG="${MODEL//\//_}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-$$}}"
RUN_DIR="${RUN_DIR:-$RESULTS_ROOT/$MODEL_TAG-$BENCHMARK/$RUN_ID}"
mkdir -p "$RUN_DIR"

printf 'Run ID: %s\n' "$RUN_ID"
printf 'Run directory: %s\n' "$RUN_DIR"

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
printf 'Server backend: %s\n' "$SERVER_BACKEND"
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

if [[ "$SERVER_BACKEND" == "vllm-pr38888-image-pruning" ]]; then
    vllm_command+=(
        --image-pruning-rate "$IMAGE_PRUNING_RATE"
        --vit-attention-score-layer-index "$VIT_ATTENTION_SCORE_LAYER_INDEX"
        --mm-encoder-attn-backend FLASH_ATTN
        --no-enable-chunked-prefill
    )

    printf '%s\n' \
        'WARNING: using an unmerged experimental vLLM PR for image-token pruning.' \
        "Image pruning rate: $IMAGE_PRUNING_RATE" \
        "ViT attention layer: $VIT_ATTENTION_SCORE_LAYER_INDEX" \
        "vLLM source commit: $IMAGE_PRUNING_VLLM_COMMIT"

    "$CONDA_ENV/bin/python" - \
        "$RUN_DIR/server-config.json" \
        "$SERVER_BACKEND" \
        "$IMAGE_PRUNING_VLLM_REPOSITORY" \
        "$IMAGE_PRUNING_VLLM_COMMIT" \
        "$IMAGE_PRUNING_VLLM_BASE_COMMIT" \
        "$VLLM_VERSION" \
        "$IMAGE_PRUNING_RATE" \
        "$VIT_ATTENTION_SCORE_LAYER_INDEX" \
        "$MODEL" \
        "$ENCODER_PATCH" \
        "${vllm_command[@]:4}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = {
    "server_backend": sys.argv[2],
    "source_repository": sys.argv[3],
    "source_commit": sys.argv[4],
    "precompiled_wheel_commit": sys.argv[5],
    "vllm_version": sys.argv[6],
    "image_pruning_rate": sys.argv[7],
    "vit_attention_score_layer_index": sys.argv[8],
    "mm_encoder_attention_backend": "FLASH_ATTN",
    "model": sys.argv[9],
    "tensor_parallel_size": 1,
    "encoder_patch": sys.argv[10],
    "server_arguments": sys.argv[11:],
}
if path.exists():
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit(f"Run configuration does not match {path}")
else:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
PY
fi

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
        if [[ "$SERVER_BACKEND" == "vllm-pr38888-image-pruning" ]]; then
            mmiu_arguments+=(--backend-signature "$BACKEND_SIGNATURE")
        fi
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
            --vendor-root "$CROSSVID_VENDOR_ROOT" \
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
