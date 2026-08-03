# Qwen3-VL Benchmarking

Reproducible runners for benchmarking Qwen3-VL models on MMIU and CrossVid
through an OpenAI-compatible vision endpoint such as vLLM. The runners adapt
the official benchmark implementations and add deterministic answer extraction,
resumable output, manifests, and strict coverage checks.

## Install

Install `uv`, then let it provision Python 3.11 and the project environment. Do
not create or activate a virtual environment manually. `.python-version` pins
the interpreter selected by uv.

```bash
uv python install 3.11
uv sync --python 3.11 --extra crossvid --group dev
```

`uv.lock` pins the resolved environment. Add `--frozen` to `uv sync` in CI or
on benchmark machines to prevent lockfile changes.

Clone the pinned official implementations. MMIU is retained as a provenance
reference; `crossvid_eval.py` directly imports CrossVid's official media
preprocessors.

```bash
mkdir -p vendor
git clone https://github.com/OpenGVLab/MMIU.git vendor/MMIU
git -C vendor/MMIU checkout 642001df618a57d65869ef3975021deabfbcc891
git clone https://github.com/chuntianli666/CrossVid.git vendor/CrossVid
git -C vendor/CrossVid checkout b53ada63551f9ac4a726b381b627d17ece066281
```

## Serve Qwen3-VL

```bash
uv sync --python 3.11 --extra crossvid --extra serve --group dev
uv run vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":128}'
```

Replace the model ID with the Qwen3-VL checkpoint being measured. Use a
separate result path for each checkpoint and keep server settings fixed when
comparing models or token-compression methods.

## MMIU Data

Download the pinned Parquet metadata and seven media archives:

```bash
uv run hf download FanqingM/MMIU-Benchmark \
  --repo-type dataset \
  --revision 03bf7d143d920e97a757f606b6b7baee161b019b \
  --local-dir data/MMIU
```

Extract every ZIP into `data/MMIU`. Paths must match `input_image_path`, for
example:

```text
data/MMIU/High-level-obj-semantic/person_reid/person_reid_0_0.jpg
```

Inspect the pinned release:

```bash
uv run mmiu-eval inspect --dataset-path data/MMIU/all.parquet
```

It contains 11,698 labeled rows and 60 stored `task` values. The paper's 52
conceptual tasks combine some source-specific stored values. This evaluator
reports all 60 released values and their unweighted macro-average rather than
inventing an undocumented grouping.

## MMIU Evaluation

Run a smoke test:

```bash
uv run mmiu-eval run \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --media-root data/MMIU \
  --dataset-path data/MMIU/all.parquet \
  --output results/qwen3-vl-8b-mmiu-smoke.jsonl \
  --limit 10
```

Use a new output path and omit `--limit` for a full run:

```bash
uv run mmiu-eval run \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --media-root data/MMIU \
  --dataset-path data/MMIU/all.parquet \
  --output results/qwen3-vl-8b-mmiu.jsonl
```

Successful rows are skipped when the command is rerun; failed rows are retried.
The adjacent manifest prevents mixing subsets, model IDs, endpoints, transport,
or generation settings. Score an existing run with:

```bash
uv run mmiu-eval score \
  --output results/qwen3-vl-8b-mmiu.jsonl \
  --strict
```

Strict scoring exits nonzero when rows are missing or API calls failed.
Unparseable answers count as incorrect. The official MMIU code uses another
language model to map verbose predictions to option letters. This runner keeps
the official task-dependent prompt ordering and macro-average, but requests and
deterministically extracts one option letter. Report this extraction difference
when comparing results with the original leaderboard.

For `--image-transport file-url`, start vLLM with a suitable
`--allowed-local-media-path`. The default `data-uri` transport needs no local
media access in the server.

## CrossVid Data

Download the pinned CrossVid release, which is about 312 GB:

```bash
uv run hf download Chuntianli/CrossVid \
  --repo-type dataset \
  --revision 4cc98eee034e6f3950c19803485402661f54c1f8 \
  --local-dir data/CrossVid
```

Due to source-video licenses, Charades and Animal Kingdom videos must be
obtained from their original repositories and merged into
`data/CrossVid/videos/behavior`. Verify all ten annotation files:

```bash
uv run crossvid-score inspect --qa-dir data/CrossVid/QA
```

The expected total is 9,015 examples.

## CrossVid Evaluation

Run one task as a smoke test:

```bash
uv run crossvid-eval run \
  --task BU \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --qa-dir data/CrossVid/QA \
  --video-root data/CrossVid/videos \
  --uav-root data/CrossVid/uav \
  --results-dir results/crossvid-smoke \
  --workers 4 \
  --frames 128 \
  --length 360 \
  --limit 10
```

Use a clean result directory for all ten tasks:

```bash
uv run crossvid-eval run \
  --task all \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --qa-dir data/CrossVid/QA \
  --video-root data/CrossVid/videos \
  --uav-root data/CrossVid/uav \
  --results-dir results/qwen3-vl-8b-crossvid \
  --workers 4 \
  --frames 128 \
  --length 360
```

The wrapper preserves CrossVid's official frame, interval, and UAV processing.
It writes append-only state under `RESULTS_DIR/.state` and exports the official
`TASK_result.json` format. Rerunning the same command skips successful examples
and retries failures. A manifest prevents mixing model, endpoint, frame count,
resolution, generation, or subset settings.

CrossVid's open-ended CCQA task requires an LLM judge. Point this command to a
fixed text-model endpoint after inference:

```bash
uv run crossvid-eval judge \
  --model Qwen/Qwen3-32B \
  --base-url http://127.0.0.1:8001/v1 \
  --qa-dir data/CrossVid/QA \
  --results-dir results/qwen3-vl-8b-crossvid \
  --workers 8
```

Judge choice affects CCQA and the overall score. Use the same judge for every
model in an experiment. The judge prompt follows the official coverage and
correctness rubric, and each record stores the judge model and raw judgment.

Compute final metrics only after all inference tasks and CCQA judging finish:

```bash
uv run crossvid-score score \
  --qa-dir data/CrossVid/QA \
  --results-dir results/qwen3-vl-8b-crossvid \
  --json-output results/qwen3-vl-8b-crossvid/summary.json
```

The strict scorer rejects missing, duplicate, and unknown IDs. It recomputes
exact matching for BU/NC/CC/PEA/PI/PSS/MSR/MOC, temporal IoU for FSA, CCQA
coverage/correctness, dimension averages, and the unweighted ten-task `O.Avg`.
Failures remain in the denominator rather than inflating scores.

## Comparing Models

Serve and evaluate checkpoints one at a time, changing both `--model` and the
result directory. Keep `--frames`, `--length`, media, decoding, and CCQA judge
fixed. For token-compression experiments, also record compression settings,
model revision, vLLM version, GPU type, memory, latency, and throughput. The
result manifests capture inference-facing settings but not custom model internals.

## Direct Bash

`scripts/run_benchmark.sh` runs the same Conda bootstrap, uv synchronization,
vLLM server, and evaluator without Slurm. Run it directly from any directory.

For MMIU, the one-command launcher also downloads the pinned dataset and
extracts its media archives before starting the benchmark:

```bash
scripts/run_mmiu.sh
```

It defaults to `Qwen/Qwen3-VL-8B-Instruct`, `data/MMIU`, and the persistent
Conda prefix `${SCRATCH}/conda-envs/qwen3vl-bench` when `SCRATCH` is set. Supply
a different model as the first argument:

```bash
scripts/run_mmiu.sh Qwen/Qwen3-VL-32B-Instruct
```

Override locations or run a short smoke test with environment variables:

```bash
CONDA_ENV=/scratch/$USER/conda-envs/qwen3vl-bench \
MMIU_ROOT=/scratch/$USER/datasets/MMIU \
MMIU_LIMIT=10 \
  scripts/run_mmiu.sh
```

The first run creates the Conda environment, installs uv from conda-forge,
synchronizes `uv.lock`, downloads MMIU, extracts it, launches vLLM, and evaluates
the model. Later runs reuse the environment and extracted-data marker while the
evaluator resumes its existing output.

vLLM startup output is streamed to the terminal and saved under the run result
directory. If startup fails, the launcher prints the last 200 log lines. Set
`STREAM_SERVER_LOGS=0` to keep startup output only in the file,
`SERVER_LOG_LINES` to change the failure excerpt, or `SERVER_START_TIMEOUT` to
change the readiness timeout in seconds.

MMIU on one visible GPU:

```bash
MMIU_ROOT=/datasets/MMIU \
  scripts/run_benchmark.sh mmiu Qwen/Qwen3-VL-8B-Instruct
```

CrossVid on four visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CROSSVID_ROOT=/datasets/CrossVid \
TENSOR_PARALLEL_SIZE=4 \
  scripts/run_benchmark.sh crossvid Qwen/Qwen3-VL-32B-Instruct
```

The first invocation creates the Conda prefix, installs uv from conda-forge,
and synchronizes the locked dependencies. Override `CONDA_ENV` to control where
that environment is stored. Set `SYNC_ENV=0,BOOTSTRAP_CONDA=0` on later runs to
skip synchronization entirely.

## Slurm

`slurm/benchmark.sbatch` starts vLLM and the evaluator in one GPU allocation.
Its defaults request one GPU, 16 CPUs, 128 GB RAM, and 48 hours. Override
cluster-specific account, partition, GPU type, GPU count, memory, and time with
normal `sbatch` flags rather than editing the script.

The remote machine only needs Conda initially. By default, the Slurm job creates
a persistent Conda environment, installs uv from conda-forge, and runs
`uv sync` on the allocated compute node. This uses no pip and requires no
activation. Submit MMIU directly with:

```bash
sbatch \
  --partition=gpu \
  --account=YOUR_ACCOUNT \
  --export=ALL,MODULES=YOUR_CONDA_MODULE,BENCHMARK=mmiu,MODEL=Qwen/Qwen3-VL-8B-Instruct,SYNC_ENV=1,BOOTSTRAP_CONDA=1 \
  slurm/benchmark.sbatch
```

To use explicit data and environment locations:

```bash
sbatch \
  --partition=gpu \
  --account=YOUR_ACCOUNT \
  --export=ALL,MODULES=YOUR_CONDA_MODULE,BENCHMARK=mmiu,MODEL=Qwen/Qwen3-VL-8B-Instruct,CONDA_ENV=/scratch/$USER/conda-envs/qwen3vl-bench,MMIU_ROOT=/datasets/MMIU,SYNC_ENV=1,BOOTSTRAP_CONDA=1 \
  slurm/benchmark.sbatch
```

Submit CrossVid on four GPUs with tensor parallelism:

```bash
sbatch \
  --partition=gpu \
  --account=YOUR_ACCOUNT \
  --gres=gpu:4 \
  --export=ALL,MODULES=YOUR_CONDA_MODULE,BENCHMARK=crossvid,MODEL=Qwen/Qwen3-VL-32B-Instruct,TENSOR_PARALLEL_SIZE=4,CONDA_ENV=/scratch/$USER/conda-envs/qwen3vl-bench,CROSSVID_ROOT=/datasets/CrossVid,SYNC_ENV=1,BOOTSTRAP_CONDA=1 \
  slurm/benchmark.sbatch
```

The first job downloads and installs the locked packages. Later jobs reuse the
same Conda prefix; leaving `SYNC_ENV=1` is safe and normally becomes a quick
consistency check. Use `SYNC_ENV=0,BOOTSTRAP_CONDA=0` only after that prefix is
known to be complete. Compute nodes must be able to reach conda-forge and the
Python package index for the first synchronization.

Set `PROJECT_DIR`, `MMIU_ROOT`, `CROSSVID_ROOT`, `RESULTS_ROOT`, `MODULES`, and
cache locations through exported environment variables when the HPC layout
differs from this repository. If Conda is only exposed as a module, include
`MODULES=YOUR_CONDA_MODULE` in `--export` or load it before submission and make
sure the `conda` executable remains exported. CrossVid aggregate scoring
additionally requires `JUDGE_MODEL`; set `JUDGE_BASE_URL` and `JUDGE_API_KEY`
when the fixed judge is served outside the allocation. Without a judge, the job
preserves all inference outputs but intentionally does not report an overall
CrossVid score.

## Tests

```bash
uv run python -m unittest -v
uv run python -m compileall -q mmiu_eval.py crossvid_eval.py crossvid_score.py
uv run ruff check .
```
