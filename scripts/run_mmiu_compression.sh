#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
METHOD="${COMPRESSION_METHOD:-${1:-}}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
SYNC_ENV="${SYNC_ENV:-1}"
BOOTSTRAP_CONDA="${BOOTSTRAP_CONDA:-1}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"
STREAM_SERVER_LOGS="${STREAM_SERVER_LOGS:-1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
LLAVA_REPOSITORY="https://github.com/haotian-liu/LLaVA.git"
LLAVA_COMMIT="c121f0432da27facab705978f83c4ada465e46fd"

case "$METHOD" in
    visionzip)
        REPOSITORY="https://github.com/JIA-Lab-research/VisionZip.git"
        COMMIT="8f86b55c6f000eb033e6912538af2dd7dcb30502"
        MODEL="liuhaotian/llava-v1.5-7b"
        MODEL_REVISION="4481d270cc22fd5c4d1bb5df129622006ccd9234"
        PARAMETERS="{\"dominant_tokens\":${VISIONZIP_DOMINANT_TOKENS:-54},\"contextual_tokens\":${VISIONZIP_CONTEXTUAL_TOKENS:-10}}"
        ;;
    hiprune)
        REPOSITORY="https://github.com/Danielement321/HiPrune.git"
        COMMIT="82781005a7e72a6be9ede58fd77473efa72b5e4f"
        MODEL="liuhaotian/llava-v1.5-7b"
        MODEL_REVISION="4481d270cc22fd5c4d1bb5df129622006ccd9234"
        PARAMETERS="{\"retained_tokens\":${HIPRUNE_RETENTION:-192},\"alpha\":${HIPRUNE_ALPHA:-0.1},\"object_layer\":${HIPRUNE_OBJECT_LAYER:-9}}"
        ;;
    cdpruner)
        REPOSITORY="https://github.com/Theia-4869/CDPruner.git"
        COMMIT="9541616c40fcd5625de1cdb8ea6c33c129eb7864"
        MODEL="liuhaotian/llava-v1.5-7b"
        MODEL_REVISION="4481d270cc22fd5c4d1bb5df129622006ccd9234"
        PARAMETERS="{\"retained_tokens\":${CDPRUNER_RETAINED_TOKENS:-64}}"
        ;;
    divprune)
        REPOSITORY="https://github.com/vbdi/divprune.git"
        COMMIT="799e2d950aa01ba7860907f5a6d86061f885dca6"
        MODEL="liuhaotian/llava-v1.5-7b"
        MODEL_REVISION="4481d270cc22fd5c4d1bb5df129622006ccd9234"
        PARAMETERS="{\"retained_ratio\":${DIVPRUNE_RETAINED_RATIO:-0.098},\"layer\":0}"
        ;;
    fastv)
        REPOSITORY="https://github.com/pkunlp-icler/FastV.git"
        COMMIT="d1659729b5bf1be225e99ee15783deeea80f63b1"
        MODEL="llava-hf/llava-1.5-7b-hf"
        MODEL_REVISION="a272c74"
        PARAMETERS="{\"pruning_layer\":${FASTV_K:-3},\"pruning_fraction\":${FASTV_R:-0.75}}"
        ;;
    *)
        printf 'Usage: %s {visionzip|hiprune|cdpruner|divprune|fastv}\n' "$0" >&2
        exit 2
        ;;
esac

if [[ "$WORKERS" != "1" ]]; then
    printf 'Official method backends require WORKERS=1 (batch size one).\n' >&2
    exit 2
fi

if [[ "${PRINT_COMPRESSION_PROFILE:-0}" == "1" ]]; then
    printf '%s\n' \
        "method=$METHOD" \
        "repository=$REPOSITORY" \
        "commit=$COMMIT" \
        "model=$MODEL" \
        "model_revision=$MODEL_REVISION" \
        "parameters=$PARAMETERS" \
        "max_images_per_example=1"
    exit 0
fi

cd "$PROJECT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-$HOME/.cache}/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/.cache}/huggingface}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${SCRATCH:-$HOME/.cache}/conda-pkgs}"
export OPENAI_API_KEY
export TOKENIZERS_PARALLELISM=false

if ! command -v conda >/dev/null 2>&1; then
    printf 'conda is required to isolate the incompatible official method environments.\n' >&2
    exit 2
fi

CONDA_ENV_DIR="${CONDA_ENV_DIR:-${SCRATCH:-$PROJECT_DIR}/conda-envs}"
CONDA_ENV="${CONDA_ENV:-$CONDA_ENV_DIR/official-$METHOD-$COMMIT}"
CONDA_PYTHON="$CONDA_ENV/bin/python"
UV_BIN="$CONDA_ENV/bin/uv"
if [[ ! -x "$CONDA_PYTHON" || ! -x "$UV_BIN" ]]; then
    if [[ "$BOOTSTRAP_CONDA" != "1" ]]; then
        printf 'Official backend environment is absent and BOOTSTRAP_CONDA=0: %s\n' "$CONDA_ENV" >&2
        exit 2
    fi
    conda create --yes --prefix "$CONDA_ENV" --channel conda-forge python=3.10 uv
    SYNC_ENV=1
fi

"$CONDA_PYTHON" - "$CONDA_ENV/.vlm-token-compression-profile" "official-$METHOD@$COMMIT" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2]
if path.exists():
    actual = path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise SystemExit(
            f"Conda environment {path.parent} belongs to {actual}, expected {expected}"
        )
else:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(expected + "\n", encoding="utf-8")
    os.replace(temporary, path)
PY

VENDOR_ROOT="${COMPRESSION_VENDOR_ROOT:-$PROJECT_DIR/vendor/compression}"
METHOD_ROOT="$VENDOR_ROOT/$METHOD"
LLAVA_ROOT="$VENDOR_ROOT/LLaVA-$LLAVA_COMMIT"

clone_pinned() {
    local repository="$1"
    local commit="$2"
    local destination="$3"
    if [[ ! -d "$destination/.git" ]]; then
        if [[ -e "$destination" ]]; then
            printf '%s exists but is not a git checkout.\n' "$destination" >&2
            exit 2
        fi
        mkdir -p "$(dirname "$destination")"
        git clone --no-checkout "$repository" "$destination"
    fi
    if [[ "$(git -C "$destination" remote get-url origin)" != "$repository" ]]; then
        printf 'Unexpected origin for %s\n' "$destination" >&2
        exit 2
    fi
    git -C "$destination" fetch --quiet origin "$commit"
    git -C "$destination" checkout --quiet --detach "$commit"
}

clone_pinned "$REPOSITORY" "$COMMIT" "$METHOD_ROOT"
if [[ "$METHOD" == "visionzip" ]]; then
    clone_pinned "$LLAVA_REPOSITORY" "$LLAVA_COMMIT" "$LLAVA_ROOT"
fi

if [[ "$SYNC_ENV" == "1" ]]; then
    "$UV_BIN" pip install --python "$CONDA_PYTHON" \
        -r backends/official_compression/requirements.txt
    "$UV_BIN" pip install --python "$CONDA_PYTHON" --no-deps --editable "$PROJECT_DIR"
    case "$METHOD" in
        visionzip)
            "$UV_BIN" pip install --python "$CONDA_PYTHON" --no-deps --editable "$LLAVA_ROOT"
            "$UV_BIN" pip install --python "$CONDA_PYTHON" --no-deps --editable "$METHOD_ROOT"
            ;;
        hiprune)
            "$UV_BIN" pip install --python "$CONDA_PYTHON" --no-deps --editable "$METHOD_ROOT/LLaVA"
            ;;
        cdpruner)
            "$UV_BIN" pip install --python "$CONDA_PYTHON" --no-deps --editable "$METHOD_ROOT"
            ;;
        divprune)
            "$UV_BIN" pip install --python "$CONDA_PYTHON" --no-deps --editable "$METHOD_ROOT/LLaVA"
            ;;
        fastv)
            "$UV_BIN" pip install --python "$CONDA_PYTHON" --no-deps --editable \
                "$METHOD_ROOT/src/FastV/llava-hf/transformers"
            ;;
    esac
fi

PARAMETERS="$($CONDA_PYTHON - "$METHOD" "$PARAMETERS" <<'PY'
import json
import sys

from compression_profiles import validated_parameters

print(json.dumps(validated_parameters(sys.argv[1], json.loads(sys.argv[2])), separators=(",", ":")))
PY
)"

MMIU_ROOT="${MMIU_ROOT:-$PROJECT_DIR/data/MMIU}"
if [[ ! -f "$MMIU_ROOT/all.parquet" ]]; then
    printf '%s\n' \
        "MMIU is not prepared at $MMIU_ROOT." \
        'Prepare it once with scripts/run_mmiu.sh, or set MMIU_ROOT.' >&2
    exit 2
fi

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
MODEL_TAG="${MODEL//\//_}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-$$}}"
RUN_DIR="${RUN_DIR:-$RESULTS_ROOT/$MODEL_TAG-mmiu-$METHOD/$RUN_ID}"
mkdir -p "$RUN_DIR"
SERVER_LOG="$RUN_DIR/official-$METHOD-${SLURM_JOB_ID:-local}.log"
API_BASE_URL="http://127.0.0.1:$PORT/v1"
BACKEND_SIGNATURE="official-$METHOD@$COMMIT;model=$MODEL;revision=$MODEL_REVISION;parameters=$PARAMETERS;single-image=true"

SERVER_CONFIG="$RUN_DIR/server-config.json"
"$CONDA_PYTHON" - \
    "$SERVER_CONFIG" "$METHOD" "$REPOSITORY" "$COMMIT" "$MODEL" \
    "$MODEL_REVISION" "$PARAMETERS" "$BACKEND_SIGNATURE" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = {
    "server_backend": "official-transformers-adapter",
    "compression_method": sys.argv[2],
    "source_repository": sys.argv[3],
    "source_commit": sys.argv[4],
    "model": sys.argv[5],
    "model_revision": None if sys.argv[6] == "none" else sys.argv[6],
    "parameters": json.loads(sys.argv[7]),
    "backend_signature": sys.argv[8],
    "benchmark_scope": {"benchmark": "mmiu", "max_images_per_example": 1},
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

server_command=(
    "$CONDA_PYTHON" official_compression_server.py
    --method "$METHOD"
    --model "$MODEL"
    --repository "$METHOD_ROOT"
    --parameters "$PARAMETERS"
    --port "$PORT"
)
server_command+=(--revision "$MODEL_REVISION")
if [[ "$METHOD" == "visionzip" ]]; then
    server_command+=(--llava-repository "$LLAVA_ROOT")
fi

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
if [[ "$STREAM_SERVER_LOGS" == "1" ]]; then
    "${server_command[@]}" > >(tee "$SERVER_LOG") 2>&1 &
else
    "${server_command[@]}" >"$SERVER_LOG" 2>&1 &
fi
SERVER_PID=$!

"$CONDA_PYTHON" - "$API_BASE_URL/models" "$SERVER_PID" "$SERVER_START_TIMEOUT" <<'PY'
import os
import sys
import time
import urllib.request

url, pid, timeout = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                raise SystemExit(0)
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit("official compression server exited before readiness")
    time.sleep(5)
raise SystemExit(f"timed out waiting for {url}")
PY

mmiu_arguments=(
    --model "$MODEL"
    --model-family generic
    --base-url "$API_BASE_URL"
    --api-key "$OPENAI_API_KEY"
    --media-root "$MMIU_ROOT"
    --dataset-path "$MMIU_ROOT/all.parquet"
    --output "$RUN_DIR/results.jsonl"
    --workers "$WORKERS"
    --max-images-per-example 1
    --backend-signature "$BACKEND_SIGNATURE"
)
if [[ -n "${MMIU_LIMIT:-}" ]]; then
    mmiu_arguments+=(--limit "$MMIU_LIMIT")
fi
if [[ -n "${MMIU_TASKS:-}" ]]; then
    mmiu_arguments+=(--tasks "$MMIU_TASKS")
fi
"$CONDA_PYTHON" -m mmiu_eval run "${mmiu_arguments[@]}"

printf 'Official %s benchmark output: %s\n' "$METHOD" "$RUN_DIR"
