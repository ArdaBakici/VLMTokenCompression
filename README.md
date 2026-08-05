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

Each ZIP must be extracted into a directory matching its archive stem. For
example, extract `High-level-sub-semantic.zip` into
`data/MMIU/High-level-sub-semantic`, not directly into `data/MMIU`. Paths must
match `input_image_path`, for example:

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

Because unparseable answers are scored as incorrect, a model that phrases its
answers differently can lose macro accuracy without answering worse. Compare the
`Invalid predictions` counts of two runs before comparing their scores, and
inspect the discarded rows with:

```bash
uv run python scripts/show_invalid.py results/Qwen_Qwen3-VL-8B-Instruct-mmiu/<run>
```

It accepts a `results.jsonl` file, a run directory, or a tree of run
directories, and reports API failures and unparseable predictions separately,
grouped by task and by reason, with the raw prediction text and the run's
`max_tokens`. Predictions that all stop near one length were truncated by
`--max-tokens` rather than malformed. Use `--task` to focus on one task,
`--limit 0` to print every row, and `--width 0` to stop truncating the text.

For `--image-transport file-url`, start vLLM with a suitable
`--allowed-local-media-path`. The default `data-uri` transport needs no local
media access in the server.

## CrossVid Data

`scripts/run_crossvid.sh` performs every step in this section and the next one.
Follow them manually only when driving the pieces separately.

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

The expected total is 9,015 examples. `scripts/prepare_crossvid.py` checks that
total along with every referenced video and the per-view UAV frame directories
and bounding-box files:

```bash
uv run python scripts/prepare_crossvid.py \
  --root data/CrossVid \
  --marker data/CrossVid/.prepared-4cc98eee034e6f3950c19803485402661f54c1f8
```

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

## Experimental Image Pruning

The repository includes an opt-in MMIU profile for the attention-based image
token pruning proposed in vLLM PR
[`#38888`](https://github.com/vllm-project/vllm/pull/38888). It prunes the
lowest-scoring image tokens before decoder fusion and supports Qwen3-VL's image
path through the existing OpenAI-compatible endpoint.

This is a controlled research profile, not released vLLM functionality. The PR
is unmerged, needs rebasing, and has no HTTP-level or tensor-parallel pruning
test. Its checked-in Qwen3-VL tests use the 2B checkpoint rather than 8B. The
implementation performs VisionZip-inspired dominant-token selection, not the
complete dominant-plus-contextual-token VisionZip algorithm. Results must be
reported as `vLLM PR #38888 image pruning`, not as upstream vLLM or full
VisionZip.

The profile runs on one GPU and accepts `Qwen/Qwen3-VL-8B-Instruct` and
`Qwen/Qwen3-VL-30B-A3B-Instruct`. Both share the 27-layer Qwen3-VL vision tower
the PR scores, so `VIT_ATTENTION_SCORE_LAYER_INDEX` selects the same layer in
each; a checkpoint with a different vision depth would silently score a
different layer, which is why the list is an allowlist rather than a warning.
The profile disables chunked prefill to avoid the M-RoPE media-boundary bug
tracked in vLLM issue
[`#48833`](https://github.com/vllm-project/vllm/issues/48833). This trades some
serving throughput for correctness on MMIU's multi-image prompts.

The launcher also applies two local source patches to the installed pinned
build, `scripts/patch_image_pruning.py`. Both are idempotent and refuse to run
if the pinned revision stops matching what they expect. Report them alongside
the PR, and record `encoder_patch` from `server-config.json` with the other
compression settings.

`sequential-image-encoding-v1` bounds vision-tower memory. vLLM charges an image
against the multimodal encoder compute budget using the number of tokens it
contributes to the prompt, which pruning reduces, while the vision tower still
runs on every unpruned patch. Disabling chunked prefill additionally raises that
budget to `--max-model-len`. Together these let the scheduler pack several whole
MMIU prompts into a single vision-tower forward, far beyond what startup memory
profiling reserved, which exhausts device memory on an 80 GB or 94 GB H100. vLLM
already encodes pruned media one item at a time for Efficient Video Sampling but
gates that path to the video modality; the patch extends the gate to images. It
changes only how many images share one vision-tower call, not the generated
output, pruning rate, context length, or scored coverage.

`moe-image-pruning-rate-v1` is what makes the MoE checkpoint usable.
`Qwen3VLMoeForConditionalGeneration.__init__` calls `super()` on the grandparent
class, skipping the dense `__init__` that assigns `image_pruning_rate`, while
inheriting every image path that reads it. Both classes share one multimodal
processor, so the prompt placeholders are already shortened by the pruning rate
before the model raises `AttributeError` on its first image. The patch assigns
the attribute the inherited code expects.

Set `IMAGE_PRUNING_ENCODER_PATCH=0` to reproduce unpatched upstream behavior.
Short or single-image prompts stay within memory on the dense checkpoint, but
MMIU's multi-image prompts do not, and the MoE checkpoint fails immediately.

Run a ten-example smoke test on one visible H100:

```bash
CUDA_VISIBLE_DEVICES=0 \
MMIU_LIMIT=10 \
  scripts/run_mmiu_image_pruning.sh
```

The default pruning rate is 30%, which is the conservative setting recommended
by the PR for OCR-heavy inputs. Override it for a separate run:

```bash
CUDA_VISIBLE_DEVICES=0 \
IMAGE_PRUNING_RATE=0.5 \
  scripts/run_mmiu_image_pruning.sh
```

Pass the MoE checkpoint as the first argument to prune it instead:

```bash
CUDA_VISIBLE_DEVICES=0 \
  scripts/run_mmiu_image_pruning.sh Qwen/Qwen3-VL-30B-A3B-Instruct
```

It needs a 94 GB card: 62.1 GB of bfloat16 weights, a 1.6 GB deepstack buffer
sized by `max_num_batched_tokens`, and 12.9 GB of KV cache for 131,072 tokens
fit the 83.8 GB that `--gpu-memory-utilization 0.90` reserves, with the vision
tower inside the profiled budget once the patches are applied. This profile
rejects tensor parallelism, so on an 80 GB card the only option is to shorten
`MAX_MODEL_LEN` and accept that longer MMIU rows are recorded as failures.

The launcher pins source commit
`d093d3037350eb7c9de1d149f9311432de2e0adb`, installs it using compatible
precompiled native extensions from upstream base commit
`4eefbf9609e5ddb996e3ac37e192e92466ec35cc`, forces the FlashAttention vision
backend, and rejects tensor parallelism. It uses a separate persistent Conda
environment named `qwen3vl-image-pruning-d093d3037`, leaving the baseline vLLM
environment untouched. The mutually exclusive baseline and image-pruning
dependency graphs are both pinned in `uv.lock`. The first run builds the pinned
source package using the compatible native wheel; later runs reuse the uv cache.
The native wheel requires an x86-64 host with glibc 2.31 or newer.

Each compressed run records `server-config.json` in its result directory.
Resuming with a different model, pruning rate, attention layer, backend, source
revision, or encoder patch is rejected by both the server configuration and MMIU
manifest.
As with the baseline, set
`SYNC_ENV=0,BOOTSTRAP_CONDA=0` only after the dedicated environment has been
installed successfully.

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
the model. Later runs reuse the environment and extracted-data marker.

Every invocation creates a unique result directory using a UTC timestamp and
the process or Slurm job ID:

```text
results/Qwen_Qwen3-VL-8B-Instruct-mmiu/20260803T142530Z-12345/
```

This prevents a new run from overwriting or appending to previous results. To
resume a specific interrupted run, explicitly pass its existing directory:

```bash
RUN_DIR=/path/to/existing/run scripts/run_mmiu.sh
```

You can also set `RUN_ID` to choose a stable name while retaining the standard
model/benchmark directory hierarchy.

The launcher validates all 79,259 referenced media files before starting vLLM.
It also repairs data extracted by older versions of this script, which placed
task directories directly under `data/MMIU`. Existing failed JSONL records are
retried automatically after the media layout is repaired.

CrossVid has the same one-command launcher. It clones the pinned upstream
checkout, downloads the pinned dataset revision, validates the annotations and
media, starts vLLM, and runs all ten tasks:

```bash
scripts/run_crossvid.sh
```

It accepts the same model argument, `RUN_DIR`, and `RUN_ID` handling as
`scripts/run_mmiu.sh`, and reads `CROSSVID_ROOT`, `CROSSVID_TASK`, `FRAMES`,
`FRAME_LENGTH`, and `WORKERS`. Set `JUDGE_MODEL` to judge CCQA and compute the
aggregate score in the same invocation:

```bash
CROSSVID_ROOT=/scratch/$USER/datasets/CrossVid \
JUDGE_MODEL=Qwen/Qwen3-32B \
JUDGE_BASE_URL=http://127.0.0.1:8001/v1 \
  scripts/run_crossvid.sh
```

Budget for the download: the pinned release is about 312 GB, and the first run
fetches all of it. It resumes if interrupted, and later runs reuse the marker in
`CROSSVID_ROOT`. `crossvid_eval.py` imports the official media preprocessors, so
the launcher also clones `vendor/CrossVid` and checks out the pinned commit when
it is absent; override the location with `CROSSVID_VENDOR_ROOT`.

Validation covers all ten annotation files, the released total of 9,015
examples, every referenced video, and the per-view UAV frame directories and
bounding-box files that MSR and MOC read. The behavior genre is the expected
gap: Charades and Animal Kingdom forbid redistribution, so those videos are
never part of the download and must be merged into `videos/behavior` from their
original repositories. The launcher refuses to start when they are absent,
because every example that reads them would fail and would still count in the
denominator of the reported score. To benchmark the remaining genres anyway:

```bash
CROSSVID_ALLOW_MISSING_BEHAVIOR=1 scripts/run_crossvid.sh
```

Report that as reduced coverage, not as a CrossVid score.

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
and synchronizes the locked dependencies. Set `SYNC_ENV=0,BOOTSTRAP_CONDA=0` on
later runs to skip synchronization entirely.

Environments live in `${SCRATCH}/conda-envs/` when `SCRATCH` is set and in
`PROJECT_DIR/conda-envs/` otherwise, under a per-backend name so the baseline
and image-pruning dependency graphs stay separate. `CONDA_ENV_DIR` moves that
whole directory. Point it at your Conda environments directory to keep the
prefixes where `conda env list` already looks:

```bash
CONDA_ENV_DIR="$HOME/.conda/envs" scripts/run_mmiu.sh
```

That creates `~/.conda/envs/qwen3vl-bench`, which conda then treats as a named
environment you can `conda activate qwen3vl-bench`.

`CONDA_ENV` still overrides the full path of a single environment. Prefer
`CONDA_ENV_DIR` when you use both server backends: one exported `CONDA_ENV`
points them at the same prefix, and each `uv sync` then replaces the other
backend's vLLM build. Expect roughly 20 GB per environment, and note that the
uv cache, Hugging Face cache, and Conda package cache are separate: they follow
`UV_CACHE_DIR`, `HF_HOME`, and `CONDA_PKGS_DIRS`, which default to `${SCRATCH}`
or `$HOME/.cache`.

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
