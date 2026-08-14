# VLM Token Compression Benchmarking

Reproducible runners for benchmarking Qwen3-VL, InternVL3, and LLaVA-NeXT models on MMIU and CrossVid
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

## Supported Models

The one-command launchers select serving and prompt profiles for these models:

| Model | Hugging Face ID | Context default | Notes |
| --- | --- | ---: | --- |
| Qwen3-VL 8B | `Qwen/Qwen3-VL-8B-Instruct` | 131,072 | Existing default; supports the Qwen thinking template option. |
| InternVL3 8B | `OpenGVLab/InternVL3-8B-hf` | 32,768 | Preferred native Transformers/vLLM checkpoint. |
| LLaVA-NeXT 7B | `llava-hf/llava-v1.6-mistral-7b-hf` | 32,768 | Mistral image checkpoint; enables multimodal-string interleaving and folds CrossVid's system instruction into the user turn. |

Use `MODEL_REVISION` to pin a checkpoint commit. Runs record model family,
revision, vLLM version, context length, tensor parallelism, dtype, image limit,
and exact server arguments in `server-config.json`; resuming with different
server settings is rejected.

The LLaVA checkpoint above is the standard image-based LLaVA-NeXT model, not
LLaVA-NeXT-Video. CrossVid continues to use the official sampled-frame pipeline
and sends each selected frame as an image.

## Serve A Model

```bash
uv sync --python 3.11 --extra crossvid --extra serve --group dev
uv run vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":128}'
```

Replace the model ID with the checkpoint being measured. Use a separate result
path for each checkpoint and keep server settings fixed when comparing models or
token-compression methods.

The documented manual server command is Qwen-specific. Prefer the launchers
below for InternVL and LLaVA because they apply the required model profile.

Run MMIU with either added model using:

```bash
scripts/run_mmiu.sh OpenGVLab/InternVL3-8B-hf
scripts/run_mmiu.sh llava-hf/llava-v1.6-mistral-7b-hf
```

Only the native `OpenGVLab/InternVL3-8B-hf` format is supported. The original
remote-code checkpoint does not ship a vLLM-compatible chat template.

### LLaVA-NeXT Context Budget

MMIU rows carry up to 62 images, and LLaVA-NeXT AnyRes expands each image to
between 1,176 and 2,928 prompt tokens depending on its pixel dimensions: a
336x336 image selects the 336x672 grid and costs 1,176 tokens, while any image
512x512 or larger selects 672x672 and costs 2,928. Image count alone therefore
does not determine whether a row fits; sixteen small images need 18,816 tokens
and fit a 32K context, while sixteen large ones need 46,848 and do not.

Before inference the evaluator reproduces the Transformers
`LlavaNextProcessor` calculation exactly, reading image headers to obtain each
row's real visual length, and adds a conservative text estimate, chat-template
overhead, and the `--max-tokens` output allowance. Rows are compared against
`--max-model-len`, which the launcher passes from the value used to start the
server. Only rows whose outcome is not already settled by the per-image bounds
are measured, so most of the dataset needs no image reads.

A run stops when selected rows do not fit, listing their measured lengths:

```bash
uv run mmiu-eval run ... --max-model-len 32768 --skip-oversized-rows
```

`--skip-oversized-rows` evaluates the rows that fit and records the decision in
the manifest. It also prints task coverage, because dropped rows can remove
whole MMIU tasks from the unweighted macro average, which then covers fewer than
60 tasks. Report such a run as reduced coverage rather than an MMIU score, and
compare only runs whose manifests list the same indices. `MMIU_SKIP_OVERSIZED=1`
sets the flag through `scripts/run_mmiu.sh`.

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

### Efficiency Metrics

New inference runs stream responses and store an `efficiency` object with every
MMIU result and internal CrossVid state record. This does not change generated
text or CrossVid's exported `TASK_result.json` schema. The following metrics are
collected per example:

| Field | Definition |
| --- | --- |
| `ttft_ms` | Client-observed time from API submission to the first non-empty generated content or reasoning event. |
| `api_request_ms` | Time until the stream, including its final usage event, is complete. |
| `e2e_generation_ms` | API submission through the terminal completion event. |
| `preprocessing_ms` | Local media/path and prompt preparation before API submission. For CrossVid this includes official video frame extraction and JPEG encoding. |
| `end_to_end_ms` | Local preprocessing, API generation, and answer parsing. Executor queue time is excluded. |
| `tpot_ms` | Time per output token after the first: `(completion event - first token) / (completion tokens - 1)`. It is absent for outputs shorter than two tokens. |
| `output_tokens_per_second` | Per-request decode rate, the inverse of TPOT. This is not aggregate server throughput. |
| `prompt_tokens` | Server-reported rendered input length, including multimodal placeholders. |
| `completion_tokens` / `total_tokens` | Server-reported output and total token counts. |
| `visual_tokens_before` / `visual_tokens_after` | Optional custom server fields at the compression boundary. Standard OpenAI and vLLM responses do not expose them. |

Durations use a monotonic clock. TTFT is client-perceived, so it includes request
transport, server queueing, multimodal processing, prefill, and first-event
delivery, but excludes local preprocessing. Report p50 and tail percentiles with
the worker count: the default four workers measure serving under concurrency,
not isolated single-request latency. Use `--workers 1` when measuring latency.
SDK retry and backoff time is included in a logical request's latency, while
usage covers only the successful response.

`scripts/show_results.sh` now includes mean prompt tokens and p50 TTFT/end-to-end
latency. Its `--json` output contains the complete mean, p50, p95, and p99
summary plus token totals. Existing runs remain readable and show `n/a` because
latency and usage cannot be reconstructed after inference. Because streaming is
part of the immutable run configuration, start a new run rather than resuming a
pre-instrumentation manifest.

Compare a compressed run against an otherwise identical baseline with:

```bash
uv run efficiency-compare \
  results/baseline/run \
  results/compressed/run
```

The comparison pairs examples by MMIU index or CrossVid task and ID, preventing
failures or subset differences from biasing token totals. If the compressed
backend reports measured visual token counts for every paired example, it reports:

- `retention_ratio = visual_tokens_after / visual_tokens_before`
- `reduction_fraction = 1 - retention_ratio`
- `compression_factor = visual_tokens_before / visual_tokens_after`

Otherwise it reports the same values for **total prompt tokens** and labels the
metric `paired_total_prompt_tokens`. This is a useful end-to-end proxy for the
current image-pruning backend, whose processor shortens multimodal placeholders,
but it is not a general visual-token count. Methods that prune inside later LLM
layers can reduce compute without changing API prompt usage. For those methods,
instrument the model at a named boundary and expose the two optional usage
fields. Always store raw before/after counts because "compression ratio" is used
inconsistently across papers.

Common additional serving metrics are aggregate requests/s, aggregate output
tokens/s, and peak GPU memory. They require a controlled timed load interval or
server/worker instrumentation and are intentionally not inferred from concurrent
per-request timings. For memory, measure post-warmup peak allocated and reserved
bytes inside every vLLM GPU worker, or sample device framebuffer use with
NVML/DCGM on dedicated GPUs; `--gpu-memory-utilization` is a configuration, not
a measured peak.

Two wrappers report on completed runs. Neither starts a server, synchronizes the
lockfile, or bootstraps Conda: they locate an environment that already has the
project installed and read result files. Set `CONDA_ENV` or `CONDA_ENV_DIR` if
your environments are somewhere unusual.

`scripts/show_results.sh` prints one row per run:

```bash
scripts/show_results.sh                     # RESULTS_ROOT, or ./results
scripts/show_results.sh path/to/results     # a tree, a run, or a results.jsonl
REPARSE=1 scripts/show_results.sh           # rescore with the current extractor
scripts/show_results.sh --json              # machine-readable
```

`scripts/score_results.sh` runs each benchmark's own scorer instead, printing the
official per-task tables:

```bash
scripts/score_results.sh                    # every run under ./results
scripts/score_results.sh path/to/run        # one run
STRICT=0 scripts/score_results.sh           # report instead of failing
```

It dispatches on what a run directory holds: `results.jsonl` is scored with
`mmiu-eval score`, `{task}_result.json` files with `crossvid-score score`, which
also refreshes that run's `summary.json`. Strict scoring is on by default, so the
script exits nonzero when any run is incomplete or had API failures, but every
run is scored first — one bad run does not hide the others.

Both wrappers call `scripts/show_runs.py`, which can also be run directly:

```bash
uv run python scripts/show_runs.py results
```

```text
benchmark  model                           profile                         rows    fail   invalid        score
mmiu       Qwen/Qwen3-VL-30B-A3B-Instruct  prune=0.3 layer=-2 glibc-shim   11698   0      1179 (10.1%)   55.10
mmiu       Qwen/Qwen3-VL-8B-Instruct       baseline                        11698   0      87 (0.7%)      58.73
```

It reads MMIU runs from their JSONL, applying the same manifest filtering strict
scoring uses, and reports CrossVid runs from the `summary.json` that
`crossvid-score score` wrote. Read `fail` and `invalid` before comparing `score`:
API failures and unparseable answers are both counted as incorrect, so either can
move the headline number without the model behaving differently, and the run with
more of them is not comparable to its neighbour.

Because answer extraction is deterministic, `--reparse` rescores stored
predictions with the current extractor and shows both numbers, without modifying
any results file:

```bash
uv run python scripts/show_runs.py results \
  --reparse --dataset-path data/MMIU/all.parquet
```

Add `--json` for machine-readable output. The `profile` column comes from each
run's `server-config.json`, so compressed runs report their pruning rate,
attention layer and any local patches alongside the score.

## Official Compression Methods

Five training-free methods can be evaluated through their pinned official model
implementations:

| Method | Official source commit | Compatible checkpoint | Default setting |
| --- | --- | --- | --- |
| VisionZip | `8f86b55c6f000eb033e6912538af2dd7dcb30502` | `liuhaotian/llava-v1.5-7b@4481d270` | 54 dominant + 10 contextual tokens |
| HiPrune | `82781005a7e72a6be9ede58fd77473efa72b5e4f` | `liuhaotian/llava-v1.5-7b@4481d270` | 192 retained, alpha 0.1, object layer 9 |
| CDPruner | `9541616c40fcd5625de1cdb8ea6c33c129eb7864` | `liuhaotian/llava-v1.5-7b@4481d270` | 64 retained tokens |
| DivPrune | `799e2d950aa01ba7860907f5a6d86061f885dca6` | `liuhaotian/llava-v1.5-7b@4481d270` | 9.8% retained at layer 0 |
| FastV | `d1659729b5bf1be225e99ee15783deeea80f63b1` | `llava-hf/llava-1.5-7b-hf@a272c74` | layer 3, 75% pruned |

Run one method with:

```bash
scripts/run_mmiu_compression.sh visionzip
scripts/run_mmiu_compression.sh hiprune
scripts/run_mmiu_compression.sh cdpruner
scripts/run_mmiu_compression.sh divprune
scripts/run_mmiu_compression.sh fastv
```

Each method receives a separate persistent Conda prefix named
`official-METHOD-COMMIT`. The launcher clones and verifies the exact source
commit, installs its incompatible legacy model fork in that prefix, starts the
shared OpenAI-compatible adapter, and runs the normal resumable MMIU evaluator.
Set `SYNC_ENV=0,BOOTSTRAP_CONDA=0` after the environment has been installed.

The released implementations are batch-one, single-image LLaVA paths. The
launcher therefore sets `--workers 1` by default and evaluates only MMIU rows
with at most one image. The filter is stored in the manifest and these results
must be reported as **single-image MMIU subset scores**, never full MMIU scores.
CrossVid is not supported because it sends multiple sampled frames. The methods
are also not relabeled as implementations for Qwen3-VL, InternVL3, or the HF
LLaVA-NeXT-Mistral checkpoint; paper-reported ports without released code are not
silently reconstructed.

Override the official settings with environment variables:

```bash
VISIONZIP_DOMINANT_TOKENS=108 VISIONZIP_CONTEXTUAL_TOKENS=20 \
  scripts/run_mmiu_compression.sh visionzip

HIPRUNE_RETENTION=128 HIPRUNE_ALPHA=0.1 HIPRUNE_OBJECT_LAYER=9 \
  scripts/run_mmiu_compression.sh hiprune

CDPRUNER_RETAINED_TOKENS=128 \
  scripts/run_mmiu_compression.sh cdpruner

DIVPRUNE_RETAINED_RATIO=0.2 \
  scripts/run_mmiu_compression.sh divprune

FASTV_K=2 FASTV_R=0.5 \
  scripts/run_mmiu_compression.sh fastv
```

Parameters are validated against each released implementation and recorded in
`server-config.json` and the evaluator backend signature. The adapter exposes
the nominal measured boundary as `visual_tokens_before` and
`visual_tokens_after`: 576 initial LLaVA-1.5 patch tokens and the configured
retained budget. FastV is different from the other four: all 576 tokens enter
the early decoder layers and only later layers see the retained count, so API
prompt usage remains uncompressed.

VisionZip, HiPrune, CDPruner, and FastV use Apache-2.0 code. DivPrune's official
repository is CC BY-NC 4.0; do not use that backend commercially. The official
repositories do not provide mutually consistent lockfiles, so the adapter pins
their source commits and central legacy requirements, and records the resulting
environment separately from the vLLM environments.

MMIU must already be prepared under `MMIU_ROOT`. Existing baseline setup does
this automatically. Use `MMIU_TASKS` and `MMIU_LIMIT` to select a smaller
single-image experiment. Result comparison still requires identical selected
indices and generation settings.

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

It needs a 94 GB card and a shorter context than the dense checkpoint. Measured
on one 93.1 GiB H100 at `--gpu-memory-utilization 0.90`, which reserves 83.8 GiB:
62.1 GiB of bfloat16 weights and 0.7 GiB of CUDA graphs leave 21 GiB, and roughly
14 GiB of that goes to the activation peak of the profiling forward pass. Only
6.8 GiB remains for KV cache, against the 12.0 GiB that one 131,072-token
sequence needs at 96 KiB per token.

Shortening the context helps twice, because disabling chunked prefill ties
`max_num_batched_tokens` to `max_model_len`, so the profiling peak shrinks along
with the per-sequence requirement:

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_MODEL_LEN=73728 \
  scripts/run_mmiu_image_pruning.sh Qwen/Qwen3-VL-30B-A3B-Instruct
```

Use the same `MAX_MODEL_LEN` for every arm being compared, including the dense
checkpoint, or the arms differ in which MMIU rows they can answer at all. Rows
whose prompt exceeds the context are rejected by the server and recorded as
failures, which strict scoring counts and reports as `API failures`; check that
count before trusting a comparison. This profile rejects tensor parallelism, so
splitting the model across GPUs is not an option here.

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

The wheel's CUDA variant is pinned to `cu129` in both `pyproject.toml` and
`IMAGE_PRUNING_WHEEL_VARIANT`, because the image-pruning extra resolves to torch
2.10, which depends on `nvidia-cuda-runtime-cu12`. Left unpinned, vLLM's
`setup.py` detects the variant from `torch.version.cuda` and falls back to
parsing `nvidia-smi` when torch is not importable, which it never is inside uv's
isolated build. On a host whose driver reports CUDA 13 that would install `cu130`
extensions beside a CUDA 12 torch. Override it only to match a different pinned
torch, and rebuild the environment when you change it.

The glibc floor is enforced per model. `vllm/_moe_C.abi3.so` imports
`log2@GLIBC_2.29`, so on a host below the wheel's manylinux_2_31 baseline it
fails to load while `vllm/_C.abi3.so` still does. vLLM logs that failed import
and carries on, so a dense checkpoint serves normally and a MoE checkpoint dies
minutes later during memory profiling with
`'_OpNamespace' '_moe_C' object has no attribute 'topk_softmax'`. The launcher
rejects a MoE checkpoint up front when `getconf GNU_LIBC_VERSION` reports less
than 2.31, and separately imports the extension after synchronizing so any other
load failure is reported before the server starts. Dense checkpoints are
unaffected.

That single symbol is the whole floor: apart from it the extension needs nothing
above glibc 2.14. `log2` has been exported as `log2@GLIBC_2.2.5` since glibc
2.2.5, and 2.29 added a faster implementation of the same function under a new
version tag. `IMAGE_PRUNING_GLIBC_SHIM=1` runs `scripts/rebind_moe_glibc.py`
against the installed extension and relaxes the launcher's glibc check:

```bash
CUDA_VISIBLE_DEVICES=0 \
IMAGE_PRUNING_GLIBC_SHIM=1 \
  scripts/run_mmiu_image_pruning.sh Qwen/Qwen3-VL-30B-A3B-Instruct
```

The script changes two bytes and needs no external tooling. It clears the
symbol's entry in `.gnu.version`, so `log2` binds to whichever implementation the
host provides, and marks the `libm.so.6` requirement in `.gnu.version_r` as
`VER_FLG_WEAK`, so the loader reports the missing version instead of refusing to
load. Both edits are required: `patchelf --clear-symbol-version` performs only
the first, and the fatal error comes from the second table, which the loader
checks in `_dl_check_map_versions` before binding any symbol. The script is
idempotent, `--verify` reports whether an extension is already rebound, and
running it against a build that does not import the symbol is a no-op.

The rebinding is recorded as `glibc_shim` in `server-config.json` and in the MMIU
backend signature. Report it with the other local modifications. The two
implementations of `log2` differ only in accuracy at the last bit and in errno
handling, and the MoE extension uses it for kernel sizing rather than for model
arithmetic, but it is a local change to a compiled artifact and belongs in the
provenance record. Running on a host that meets the baseline, or in a container,
avoids it entirely.

Each compressed run records `server-config.json` in its result directory.
Resuming with a different model, pruning rate, attention layer, backend, source
revision, wheel variant, or encoder patch is rejected by both the server
configuration and MMIU manifest.
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
Conda prefix `${SCRATCH}/conda-envs/vlm-token-compression-bench` when `SCRATCH`
is set. Supply
a different model as the first argument:

```bash
scripts/run_mmiu.sh Qwen/Qwen3-VL-32B-Instruct
```

Override locations or run a short smoke test with environment variables:

```bash
CONDA_ENV=/scratch/$USER/conda-envs/vlm-token-compression-bench \
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

Run the added models with:

```bash
scripts/run_crossvid.sh OpenGVLab/InternVL3-8B-hf
scripts/run_crossvid.sh llava-hf/llava-v1.6-mistral-7b-hf
```

Qwen retains the official runner's 128-frame default. InternVL3 defaults to 16
frames and LLaVA-NeXT to 8 because AnyRes can use substantially more visual
tokens per image, while both models have a 32K text context. CrossVid InternVL
runs additionally restrict dynamic tiling to one patch per sampled frame. The
same one-patch InternVL profile is used for MMIU so high image-count rows fit the
context; this processor setting is recorded with the run.
Override `FRAMES` only deliberately and keep it fixed across runs being compared;
the value is recorded in each task manifest. `LIMIT_MM_IMAGES` changes the vLLM
media-count allowance but does not make an oversized visual prompt fit the
context.

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

That creates `~/.conda/envs/vlm-token-compression-bench`, which conda then treats
as a named environment you can `conda activate vlm-token-compression-bench`.

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
  --export=ALL,MODULES=YOUR_CONDA_MODULE,BENCHMARK=mmiu,MODEL=Qwen/Qwen3-VL-8B-Instruct,CONDA_ENV=/scratch/$USER/conda-envs/vlm-token-compression-bench,MMIU_ROOT=/datasets/MMIU,SYNC_ENV=1,BOOTSTRAP_CONDA=1 \
  slurm/benchmark.sbatch
```

Submit CrossVid on four GPUs with tensor parallelism:

```bash
sbatch \
  --partition=gpu \
  --account=YOUR_ACCOUNT \
  --gres=gpu:4 \
  --export=ALL,MODULES=YOUR_CONDA_MODULE,BENCHMARK=crossvid,MODEL=Qwen/Qwen3-VL-32B-Instruct,TENSOR_PARALLEL_SIZE=4,CONDA_ENV=/scratch/$USER/conda-envs/vlm-token-compression-bench,CROSSVID_ROOT=/datasets/CrossVid,SYNC_ENV=1,BOOTSTRAP_CONDA=1 \
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
