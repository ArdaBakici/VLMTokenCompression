# Sourced by the reporting scripts. Finds an interpreter that already has this
# project installed, without bootstrapping Conda or synchronizing the lockfile:
# scoring reads result files and never contacts a server, so it must not pay the
# cost of a benchmark launch.

resolve_project_python() {
    local candidates=()
    if [[ -n "${CONDA_ENV:-}" ]]; then
        candidates+=("$CONDA_ENV/bin/python")
    fi
    local environment_dir="${CONDA_ENV_DIR:-${SCRATCH:-$PROJECT_DIR}/conda-envs}"
    candidates+=(
        "$environment_dir/vlm-token-compression-bench/bin/python"
        "$environment_dir/qwen3vl-bench/bin/python"
        "$environment_dir/qwen3vl-image-pruning-d093d3037/bin/python"
        "$HOME/.conda/envs/vlm-token-compression-bench/bin/python"
        "$HOME/.conda/envs/qwen3vl-bench/bin/python"
        "$HOME/.conda/envs/qwen3vl-image-pruning-d093d3037/bin/python"
        "$PROJECT_DIR/.venv/bin/python"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    printf 'No project environment found. Looked for:\n' >&2
    printf '  %s\n' "${candidates[@]}" >&2
    printf '%s\n' \
        'Set CONDA_ENV to an environment that has this project installed, or' \
        'CONDA_ENV_DIR to the directory holding the benchmark environments.' >&2
    return 2
}
