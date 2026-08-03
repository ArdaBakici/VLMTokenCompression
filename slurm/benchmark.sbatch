#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4x
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:TERM@120

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PROJECT_DIR
exec bash "$PROJECT_DIR/scripts/run_benchmark.sh"
