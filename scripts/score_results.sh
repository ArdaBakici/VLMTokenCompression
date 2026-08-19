#!/usr/bin/env bash

# Run each benchmark's own scorer over completed runs, printing the official
# per-task tables rather than the one-line-per-run summary of show_results.sh.
#
#   scripts/score_results.sh                      # every run under ./results
#   scripts/score_results.sh path/to/run          # one run
#   STRICT=0 scripts/score_results.sh             # report instead of failing
#
# MMIU runs are scored with `mmiu-eval score`, CrossVid runs with
# `crossvid-score score`. Strict scoring exits nonzero when a run is incomplete
# or had API failures; every run is still scored before this script exits, so one
# bad run does not hide the others.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
# shellcheck source=scripts/_environment.sh
source "$SCRIPT_DIR/_environment.sh"

cd "$PROJECT_DIR"
PYTHON="$(resolve_project_python)"

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/results}"
CROSSVID_ROOT="${CROSSVID_ROOT:-$PROJECT_DIR/data/CrossVid}"

runs=()
if [[ $# -gt 0 ]]; then
    runs=("$@")
else
    if [[ ! -d "$RESULTS_ROOT" ]]; then
        printf 'No results directory at %s\n' "$RESULTS_ROOT" >&2
        exit 2
    fi
    while IFS= read -r path; do
        runs+=("$path")
    done < <(find "$RESULTS_ROOT" -name results.jsonl -printf '%h\n' | sort -u)
    # crossvid_score reads RESULTS_DIR/{task}_result.json, so that is what marks
    # a CrossVid run. One directory holds up to ten of them.
    while IFS= read -r path; do
        runs+=("$path")
    done < <(find "$RESULTS_ROOT" -name '*_result.json' -printf '%h\n' | sort -u)
fi

if [[ ${#runs[@]} -eq 0 ]]; then
    printf 'No completed runs found under %s\n' "$RESULTS_ROOT" >&2
    exit 2
fi

strict_arguments=()
if [[ "${STRICT:-1}" == "1" ]]; then
    strict_arguments=(--strict)
fi

failed=()
for run in "${runs[@]}"; do
    printf '\n=== %s ===\n' "$run"
    if [[ -f "$run/results.jsonl" ]]; then
        if ! "$PYTHON" -m mmiu_eval score \
            --output "$run/results.jsonl" \
            "${strict_arguments[@]}"; then
            failed+=("$run")
        fi
    elif compgen -G "$run/*_result.json" >/dev/null; then
        if [[ ! -d "$CROSSVID_ROOT/QA" ]]; then
            printf '%s\n' \
                "Skipping: CrossVid scoring needs $CROSSVID_ROOT/QA." \
                'Set CROSSVID_ROOT to the prepared dataset directory.' >&2
            failed+=("$run")
            continue
        fi
        if ! "$PYTHON" -m crossvid_score score \
            --qa-dir "$CROSSVID_ROOT/QA" \
            --results-dir "$run" \
            --json-output "$run/summary.json"; then
            failed+=("$run")
        fi
    else
        printf 'Skipping: no results.jsonl and no {task}_result.json.\n' >&2
        failed+=("$run")
    fi
done

printf '\nScored %s run(s); %s failed.\n' "${#runs[@]}" "${#failed[@]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
