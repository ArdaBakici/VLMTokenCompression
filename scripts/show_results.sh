#!/usr/bin/env bash

# One table for every completed run under a results tree.
#
#   scripts/show_results.sh                    # RESULTS_ROOT, or ./results
#   scripts/show_results.sh path/to/results
#   REPARSE=1 scripts/show_results.sh          # rescore with the current extractor
#   scripts/show_results.sh --json             # machine-readable
#
# Any further arguments are passed through to scripts/show_runs.py.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
# shellcheck source=scripts/_environment.sh
source "$SCRIPT_DIR/_environment.sh"

cd "$PROJECT_DIR"
PYTHON="$(resolve_project_python)"

RESULTS="${RESULTS_ROOT:-$PROJECT_DIR/results}"
if [[ $# -gt 0 && "$1" != -* ]]; then
    RESULTS="$1"
    shift
fi

arguments=("$RESULTS")
if [[ "${REPARSE:-0}" == "1" ]]; then
    MMIU_ROOT="${MMIU_ROOT:-$PROJECT_DIR/data/MMIU}"
    dataset="$MMIU_ROOT/all.parquet"
    if [[ ! -f "$dataset" ]]; then
        printf '%s\n' \
            "REPARSE=1 needs the MMIU metadata, but $dataset is absent." \
            'Set MMIU_ROOT to the prepared dataset directory.' >&2
        exit 2
    fi
    arguments+=(--reparse --dataset-path "$dataset")
fi

exec "$PYTHON" scripts/show_runs.py "${arguments[@]}" "$@"
