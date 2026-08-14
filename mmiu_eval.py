"""Run and score MMIU through an OpenAI-compatible vision endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import statistics
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from efficiency_metrics import (
    EFFICIENCY_SCHEMA_VERSION,
    format_efficiency_summary,
    stream_chat_completion,
    summarize_efficiency,
)
from model_profiles import (
    MODEL_FAMILIES,
    chat_template_extra_body,
    llava_next_image_tokens,
    resolve_model_family,
)

DATASET_ID = "FanqingM/MMIU-Benchmark"
DATASET_REVISION = "03bf7d143d920e97a757f606b6b7baee161b019b"
# Bounds on one LLaVA-NeXT image, used only to skip measuring rows whose answer
# is already certain. The maximum is the 672x672 grid with no unpadding; the
# minimum is the base image alone, because unpadded and newline features are
# never negative. Both are asserted against the exact formula in the tests.
LLAVA_NEXT_MAX_IMAGE_TOKENS = 2928
LLAVA_NEXT_MIN_IMAGE_TOKENS = 577
# MMIU prompt text is estimated conservatively: English averages closer to four
# characters per token, so three overestimates the text and never lets an
# oversized row through.
TEXT_CHARS_PER_TOKEN = 3
# Chat template scaffolding around the single user turn.
PROMPT_OVERHEAD_TOKENS = 48

# The official inference script puts the question before the context for these tasks.
QUESTION_FIRST_TASKS = {
    "person_reid",
    "multiple_image_captioning",
    "spot_the_similarity",
    "face_retrieval",
    "sketch2image_retrieval",
    "handwritten_retrieval",
    "spot_the_diff",
    "image2image_retrieval",
    "vehicle_retrieval",
    "text2image_retrieval",
    "general_action_recognition",
    "video_captioning",
    "next_img_prediction",
    "temporal_ordering",
    "meme_vedio_understanding",
    "action_quality_assessment",
    "temporal_localization",
    "mevis",
    "ravens_progressive_matrices",
    "threed_indoor_recognition",
    "point_tracking",
    "threed_cad_recognition",
    "single_object_tracking",
}


def load_mmiu(dataset_path: str | None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependencies with: uv sync --extra crossvid") from exc

    if dataset_path:
        return load_dataset("parquet", data_files=dataset_path, split="train")
    return load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="test",
    )


def option_labels(options: str) -> set[str]:
    labels = set(re.findall(r"(?m)^\s*([A-N])\s*[:.)]", options.upper()))
    return labels or set("ABCD")


def parse_choice(prediction: str, valid_labels: set[str]) -> str | None:
    text = prediction.strip()
    direct = re.fullmatch(
        r"[\s\[\](){}.,:;*-]*([A-N])[\s\[\](){}.,:;*-]*", text, re.IGNORECASE
    )
    if direct and direct.group(1).upper() in valid_labels:
        return direct.group(1).upper()

    # The delimiter class must stay a single class: writing it as [\])].,:;]
    # closes after ")" and stops "B. Yes", the most common option format, from
    # ever matching.
    patterns = (
        r"^\s*(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s*[\[(]?([A-N])\b",
        r"^\s*[\[(]?([A-N])(?=[.,:;)\]]|\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1).upper() in valid_labels:
            return match.group(1).upper()
    return None


def build_prompt(row: dict[str, Any]) -> str:
    question = (row.get("question") or "").strip()
    context = (row.get("context") or "").strip()
    sections = (
        (question, context)
        if row["task"] in QUESTION_FIRST_TASKS
        else (context, question)
    )
    prompt = "\n".join(section for section in sections if section)
    return (
        f"{prompt}\nPlease answer with only the option letter, such as A, B, C, or D."
    )


def resolve_image(media_root: Path, stored_path: str) -> Path:
    relative = stored_path
    while relative.startswith("./"):
        relative = relative[2:]
    if Path(relative).is_absolute():
        raise ValueError(f"absolute dataset image path is not allowed: {stored_path}")

    root = media_root.resolve()
    image_path = (root / relative).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"dataset image path escapes media root: {stored_path}"
        ) from exc
    if not image_path.is_file():
        raise FileNotFoundError(f"missing image: {image_path}")
    return image_path


def image_url(image_path: Path, transport: str) -> str:
    if transport == "file-url":
        return image_path.as_uri()
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def infer_one(
    index: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    client: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    api_started = None
    base = {
        "index": index,
        "task": row["task"],
        "ground_truth": row.get("output"),
        "model": args.model,
        "dataset_revision": DATASET_REVISION,
        "num_images": len(row["input_image_path"]),
    }
    try:
        paths = [
            resolve_image(args.media_root, path) for path in row["input_image_path"]
        ]
        content = [
            {
                "type": "image_url",
                "image_url": {"url": image_url(path, args.image_transport)},
            }
            for path in paths
        ]
        content.append({"type": "text", "text": build_prompt(row)})
        request: dict[str, Any] = {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        model_family = resolve_model_family(
            args.model, getattr(args, "model_family", "auto")
        )
        extra_body = chat_template_extra_body(model_family, args.enable_thinking)
        if extra_body is not None:
            request["extra_body"] = extra_body
        preprocessing_ms = 1000 * (time.perf_counter() - started)
        api_started = time.perf_counter()
        prediction, efficiency = stream_chat_completion(client, request)
        parsed = parse_choice(prediction, option_labels(row.get("options") or ""))
        efficiency["preprocessing_ms"] = preprocessing_ms
        efficiency["end_to_end_ms"] = 1000 * (time.perf_counter() - started)
        return {
            **base,
            "success": True,
            "prediction": prediction,
            "choice": parsed,
            "error": None,
            "efficiency": efficiency,
        }
    except Exception as exc:  # noqa: BLE001 - persist all per-row endpoint and media failures.
        return {
            **base,
            "success": False,
            "prediction": None,
            "choice": None,
            "error": f"{type(exc).__name__}: {exc}",
            "efficiency": {
                "schema_version": EFFICIENCY_SCHEMA_VERSION,
                "preprocessing_ms": (
                    1000 * (api_started - started) if api_started is not None else None
                ),
                "api_request_ms": (
                    1000 * (time.perf_counter() - api_started)
                    if api_started is not None
                    else None
                ),
                "end_to_end_ms": 1000 * (time.perf_counter() - started),
            },
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return records


def latest_by_index(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    latest = {}
    for record in records:
        latest[int(record["index"])] = record
    return latest


def selected_indices(dataset: Any, args: argparse.Namespace) -> list[int]:
    allowed_tasks = (
        {task.strip() for task in args.tasks.split(",") if task.strip()}
        if args.tasks
        else None
    )
    indices = range(args.start, len(dataset))
    selected = [
        index
        for index in indices
        if (not allowed_tasks or dataset[index]["task"] in allowed_tasks)
        and (
            getattr(args, "max_images_per_example", None) is None
            or len(dataset[index]["input_image_path"]) <= args.max_images_per_example
        )
    ]
    if allowed_tasks and not selected:
        raise SystemExit("--tasks did not match any stored task values")
    return selected[: args.limit] if args.limit is not None else selected


def image_size(path: Path) -> tuple[int, int]:
    """Return (height, width) by reading the image header only."""

    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return height, width


def row_prompt_tokens(row: dict[str, Any], media_root: Path, max_tokens: int) -> int:
    """Exact LLaVA-NeXT prompt length for one MMIU row."""

    visual = sum(
        llava_next_image_tokens(*image_size(resolve_image(media_root, stored)))
        for stored in row["input_image_path"]
    )
    text = -(-len(build_prompt(row)) // TEXT_CHARS_PER_TOKEN)
    return visual + text + PROMPT_OVERHEAD_TOKENS + max_tokens


def partition_by_context(
    dataset: Any, indices: list[int], args: argparse.Namespace
) -> tuple[list[int], list[tuple[int, int]]]:
    """Split selected rows into those that fit the context and those that do not.

    Rows whose outcome is already decided by the per-image bounds are not
    measured, so only genuinely ambiguous rows read image headers.
    """

    if args.model_family != "llava-next":
        return indices, []

    fitting = []
    oversized = []
    for index in indices:
        row = dataset[index]
        count = len(row["input_image_path"])
        fixed = (
            -(-len(build_prompt(row)) // TEXT_CHARS_PER_TOKEN)
            + PROMPT_OVERHEAD_TOKENS
            + args.max_tokens
        )
        if count * LLAVA_NEXT_MAX_IMAGE_TOKENS + fixed <= args.max_model_len:
            fitting.append(index)
            continue
        if count * LLAVA_NEXT_MIN_IMAGE_TOKENS + fixed > args.max_model_len:
            oversized.append((index, count * LLAVA_NEXT_MIN_IMAGE_TOKENS + fixed))
            continue
        tokens = row_prompt_tokens(row, args.media_root, args.max_tokens)
        if tokens <= args.max_model_len:
            fitting.append(index)
        else:
            oversized.append((index, tokens))
    return fitting, oversized


def report_coverage(
    dataset: Any, kept: list[int], oversized: list[tuple[int, int]]
) -> None:
    """Explain which tasks a context-filtered run can still score."""

    kept_by_task: dict[str, int] = defaultdict(int)
    dropped_by_task: dict[str, int] = defaultdict(int)
    for index in kept:
        kept_by_task[dataset[index]["task"]] += 1
    for index, _ in oversized:
        dropped_by_task[dataset[index]["task"]] += 1

    tasks = set(kept_by_task) | set(dropped_by_task)
    complete = sum(1 for task in tasks if not dropped_by_task[task])
    partial = sum(1 for task in tasks if kept_by_task[task] and dropped_by_task[task])
    lost = sum(1 for task in tasks if not kept_by_task[task])
    total = len(kept) + len(oversized)
    print(
        f"Context coverage: {len(kept)}/{total} rows fit | tasks complete={complete} "
        f"partial={partial} dropped={lost}"
    )
    examples = ", ".join(
        f"{index} ({len(dataset[index]['input_image_path'])} images, ~{tokens} tokens)"
        for index, tokens in oversized[:5]
    )
    print(f"Rows exceeding the context: {len(oversized)}; first: {examples}")
    if lost or partial:
        print(
            "The macro average will cover fewer tasks than full MMIU. Report this "
            "as reduced coverage, not as an MMIU score."
        )


def manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.manifest.json")


def ensure_manifest(output: Path, expected: dict[str, Any]) -> None:
    path = manifest_path(output)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit(
                f"Run configuration does not match existing manifest: {path}"
            )
        return
    if output.exists() and output.stat().st_size:
        raise SystemExit(f"Refusing to resume {output} without its manifest")
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install dependencies with: uv sync --extra crossvid") from exc

    args.model_family = resolve_model_family(args.model, args.model_family)
    try:
        chat_template_extra_body(args.model_family, args.enable_thinking)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = load_mmiu(args.dataset_path)
    indices = selected_indices(dataset, args)
    indices, oversized = partition_by_context(dataset, indices, args)
    if oversized:
        report_coverage(dataset, indices, oversized)
        if not args.skip_oversized_rows:
            raise SystemExit(
                f"{len(oversized)} selected rows exceed --max-model-len "
                f"{args.max_model_len} for this checkpoint. Rerun with "
                "--skip-oversized-rows to evaluate the rows that fit and record "
                "the reduced coverage in the manifest, raise --max-model-len if "
                "the server allows it, or narrow the subset with "
                "--start/--limit/--tasks."
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_path": str(Path(args.dataset_path).resolve())
        if args.dataset_path
        else None,
        "model": args.model,
        "model_family": args.model_family,
        "base_url": args.base_url,
        "image_transport": args.image_transport,
        "enable_thinking": args.enable_thinking,
        "max_tokens": args.max_tokens,
        "max_images_per_example": args.max_images_per_example,
        "max_model_len": args.max_model_len,
        "skip_oversized_rows": args.skip_oversized_rows,
        "stream": True,
        "efficiency_schema_version": EFFICIENCY_SCHEMA_VERSION,
        "workers": args.workers,
        "timeout": args.timeout,
        "retries": args.retries,
        "indices": indices,
    }
    if args.backend_signature is not None:
        manifest["backend_signature"] = args.backend_signature
    ensure_manifest(args.output, manifest)

    existing = latest_by_index(read_jsonl(args.output))
    for record in existing.values():
        if (
            record.get("model") != args.model
            or record.get("dataset_revision") != DATASET_REVISION
        ):
            raise SystemExit(f"Existing results in {args.output} belong to another run")
    pending = [index for index in indices if not existing.get(index, {}).get("success")]
    if not pending:
        print(f"No pending examples; {len(indices)} already completed.")
        print_score(args.output, strict=True)
        return

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.retries,
    )
    write_lock = threading.Lock()
    completed = len(indices) - len(pending)
    failures = 0

    def persist(record: dict[str, Any]) -> None:
        nonlocal completed, failures
        with write_lock, args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            completed += 1
            failures += not record["success"]
            print(
                f"[{completed}/{len(indices)}] index={record['index']} "
                f"task={record['task']} choice={record['choice']} failures={failures}",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(infer_one, index, dataset[index], args, client): index
            for index in pending
        }
        for future in as_completed(futures):
            persist(future.result())

    print_score(args.output, strict=True)


def score_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    latest = latest_by_index(records)
    task_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    failures = 0
    invalid = 0
    unlabeled = 0
    for record in latest.values():
        if not record.get("success"):
            failures += 1
            continue
        ground_truth = record.get("ground_truth")
        if ground_truth is None:
            unlabeled += 1
            continue
        choice = record.get("choice")
        invalid += choice is None
        task_counts[record["task"]][1] += 1
        task_counts[record["task"]][0] += (
            choice is not None and choice.lower() in ground_truth.strip().lower()
        )

    per_task = {
        task: {
            "correct": counts[0],
            "total": counts[1],
            "accuracy": counts[0] / counts[1],
        }
        for task, counts in sorted(task_counts.items())
        if counts[1]
    }
    macro = (
        statistics.fmean(item["accuracy"] for item in per_task.values())
        if per_task
        else 0.0
    )
    return {
        "unique_records": len(latest),
        "failures": failures,
        "invalid_predictions": invalid,
        "unlabeled": unlabeled,
        "task_count": len(per_task),
        "macro_accuracy": macro,
        "per_task": per_task,
        "efficiency": summarize_efficiency(list(latest.values())),
    }


def reparse_records(
    records: list[dict[str, Any]], dataset: Any
) -> list[dict[str, Any]]:
    """Recompute choices from stored predictions with the current extractor.

    Answer extraction is deterministic, so a fixed extractor can be applied to
    predictions that are already stored instead of running inference again. The
    stored records are never modified: the caller scores the returned copies.
    """

    reparsed = []
    for record in records:
        prediction = record.get("prediction")
        if not record.get("success") or prediction is None:
            reparsed.append(record)
            continue
        index = int(record["index"])
        if index >= len(dataset):
            raise SystemExit(
                f"Record index {index} is outside the dataset; the results and "
                "the dataset revision do not match"
            )
        row = dataset[index]
        if row["task"] != record["task"]:
            raise SystemExit(
                f"Record index {index} is task {record['task']!r} but the dataset "
                f"holds {row['task']!r}; the results belong to another dataset"
            )
        labels = option_labels(row.get("options") or "")
        reparsed.append({**record, "choice": parse_choice(prediction, labels)})
    return reparsed


def print_score(output: Path, strict: bool, dataset: Any = None) -> None:
    records = read_jsonl(output)
    manifest_file = manifest_path(output)
    missing = 0
    unexpected = 0
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        expected = set(manifest["indices"])
        latest = latest_by_index(records)
        missing = len(expected - set(latest))
        unexpected = len(set(latest) - expected)
        records = [record for index, record in latest.items() if index in expected]
    elif strict:
        raise SystemExit(f"Strict scoring requires the run manifest: {manifest_file}")
    if dataset is not None:
        stored = score_records(records)
        records = reparse_records(records, dataset)
        print(
            "Re-parsed stored predictions with the current extractor. "
            f"Recorded macro accuracy was {stored['macro_accuracy'] * 100:.4f} with "
            f"{stored['invalid_predictions']} invalid predictions."
        )
    score = score_records(records)

    print("task,correct,total,accuracy")
    for task, item in score["per_task"].items():
        print(f"{task},{item['correct']},{item['total']},{item['accuracy'] * 100:.4f}")
    print(f"Macro accuracy: {score['macro_accuracy'] * 100:.4f}")
    print(
        f"Tasks: {score['task_count']} | Records: {score['unique_records']} | Missing: {missing} | "
        f"Unexpected: {unexpected} | "
        f"API failures: {score['failures']} | Invalid predictions: {score['invalid_predictions']}"
    )
    efficiency = score["efficiency"]
    if efficiency["measured_records"]:
        print(format_efficiency_summary(efficiency))
    if strict and (missing or unexpected or score["failures"]):
        raise SystemExit(
            "Strict scoring failed because the run is invalid or incomplete"
        )


def score_command(args: argparse.Namespace) -> None:
    dataset = load_mmiu(args.dataset_path) if args.reparse else None
    print_score(args.output, args.strict, dataset)


def inspect_dataset(args: argparse.Namespace) -> None:
    dataset = load_mmiu(args.dataset_path)
    task_counts: dict[str, int] = defaultdict(int)
    image_counts = []
    missing_labels = 0
    for row in dataset:
        task_counts[row["task"]] += 1
        image_counts.append(len(row["input_image_path"]))
        missing_labels += row.get("output") is None
    print(f"Revision: {DATASET_REVISION}")
    print(f"Rows: {len(dataset)}")
    print(f"Distinct task values: {len(task_counts)}")
    print(f"Images per row: min={min(image_counts)} max={max(image_counts)}")
    print(f"Rows without labels: {missing_labels}")
    print("task,rows")
    for task, count in sorted(task_counts.items()):
        print(f"{task},{count}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect the pinned dataset")
    inspect_parser.add_argument(
        "--dataset-path", help="optional local all.parquet path"
    )
    inspect_parser.set_defaults(function=inspect_dataset)

    run_parser = subparsers.add_parser(
        "run", help="run model inference and strict scoring"
    )
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--model-family", choices=MODEL_FAMILIES, default="auto")
    run_parser.add_argument("--media-root", required=True, type=Path)
    run_parser.add_argument("--output", required=True, type=Path)
    run_parser.add_argument("--dataset-path", help="optional local all.parquet path")
    run_parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    run_parser.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY")
    )
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--timeout", type=float, default=300)
    run_parser.add_argument("--retries", type=int, default=2)
    run_parser.add_argument("--max-tokens", type=int, default=16)
    run_parser.add_argument("--start", type=int, default=0)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--max-images-per-example", type=int)
    run_parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Server context length used to decide whether a row's prompt fits.",
    )
    run_parser.add_argument(
        "--skip-oversized-rows",
        action="store_true",
        help="Evaluate only the rows that fit the context and record the "
        "reduced coverage in the manifest.",
    )
    run_parser.add_argument("--tasks", help="comma-separated task names")
    run_parser.add_argument(
        "--image-transport", choices=("data-uri", "file-url"), default="data-uri"
    )
    run_parser.add_argument("--enable-thinking", action="store_true")
    run_parser.add_argument("--backend-signature")
    run_parser.set_defaults(function=run)

    score_parser = subparsers.add_parser("score", help="score an existing JSONL result")
    score_parser.add_argument("--output", required=True, type=Path)
    score_parser.add_argument("--strict", action="store_true")
    score_parser.add_argument(
        "--reparse",
        action="store_true",
        help="rescore stored predictions with the current extractor, without "
        "modifying the results file",
    )
    score_parser.add_argument("--dataset-path", help="optional local all.parquet path")
    score_parser.set_defaults(function=score_command)
    return root


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "workers", 1) < 1:
        raise SystemExit("--workers must be at least 1")
    if getattr(args, "start", 0) < 0:
        raise SystemExit("--start must not be negative")
    if getattr(args, "limit", 0) is not None and getattr(args, "limit", 0) < 0:
        raise SystemExit("--limit must not be negative")
    if getattr(args, "max_tokens", 1) < 1:
        raise SystemExit("--max-tokens must be at least 1")
    if (
        getattr(args, "max_images_per_example", None) is not None
        and args.max_images_per_example < 1
    ):
        raise SystemExit("--max-images-per-example must be at least 1")
    if getattr(args, "max_model_len", 1) < 1:
        raise SystemExit("--max-model-len must be at least 1")
    args.function(args)


if __name__ == "__main__":
    main()
