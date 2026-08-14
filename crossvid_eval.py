"""Run CrossVid with its official preprocessing and a Qwen3-VL endpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

from crossvid_score import TASKS, index_by_id, read_list
from efficiency_metrics import (
    EFFICIENCY_SCHEMA_VERSION,
    format_efficiency_summary,
    stream_chat_completion,
    summarize_efficiency,
)
from model_profiles import (
    MODEL_FAMILIES,
    adapt_messages,
    chat_template_extra_body,
    resolve_model_family,
)

DATASET_ID = "Chuntianli/CrossVid"
DATASET_REVISION = "4cc98eee034e6f3950c19803485402661f54c1f8"
CROSSVID_REVISION = "b53ada63551f9ac4a726b381b627d17ece066281"
MULTIPLE_CHOICE_TASKS = {"BU", "NC", "CC", "PEA", "PI", "MSR", "MOC"}

JUDGE_PROMPT = """You are asked to score the output of a model, given the following information:
- Question: {question}
- Standard Answer: {answer}
- Scoring Points: {points}
- Model's Output: {output}

Please perform the following two-part scoring:
Part 1: Coverage of Scoring Points
- For each scoring point, determine whether it is covered by the Model's Output.
- Mark as covered (true) only if the scoring point is addressed explicitly, independently, and clearly.
- If the mention is vague, partial, or ambiguous, consider it not covered.

Part 2: Accuracy of Details
- For each covered scoring point, compare the details in the Model's Output to the Standard Answer.
- Mark as correct (true) only if the details are fully accurate and consistent with the Standard Answer.
- For scoring points not covered, mark as incorrect.

Return only a JSON object with this exact shape:
{{"coverage": [true, false], "correctness": [true, false]}}
The lengths of both arrays must equal the number of scoring points.
"""


def load_official_module(task: str, vendor_root: Path) -> ModuleType:
    vendor_root = vendor_root.resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Cannot verify the CrossVid checkout at {vendor_root}") from exc
    if revision != CROSSVID_REVISION:
        raise SystemExit(
            f"CrossVid checkout is {revision}, expected {CROSSVID_REVISION}. "
            f"Run: git -C {vendor_root} checkout {CROSSVID_REVISION}"
        )

    eval_dir = vendor_root / "eval"
    module_path = eval_dir / f"{task}.py"
    if not module_path.is_file():
        raise SystemExit(
            f"Missing official CrossVid evaluator: {module_path}\n"
            "Clone it as documented in README.md or pass --vendor-root."
        )
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    spec = importlib.util.spec_from_file_location(f"_crossvid_{task}", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing CrossVid dependency {exc.name!r}; install with: "
            "uv sync --extra crossvid"
        ) from exc
    return module


def option_labels(options: list[str]) -> list[str]:
    labels = []
    for option in options:
        match = re.match(r"\s*([A-N])\s*[.) :]", option, re.IGNORECASE)
        if match:
            labels.append(match.group(1).upper())
    return labels or list("ABCD")


def answer_fragment(text: str) -> str:
    tagged = re.search(
        r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL
    )
    lines = text.strip().splitlines()
    fragment = tagged.group(1) if tagged else (lines[0] if lines else "")
    return re.sub(
        r"^\s*(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s*",
        "",
        fragment,
        flags=re.IGNORECASE,
    ).strip()


def parse_choices(prediction: str, labels: list[str], multiple: bool) -> str | None:
    fragment = answer_fragment(prediction).upper()
    if multiple:
        compact = re.sub(r"[\s,;+&/\[\](){}.-]|\bAND\b", "", fragment)
        if compact and all(letter in labels for letter in compact):
            selected = set(compact)
            return "".join(label for label in labels if label in selected)
        return None
    match = re.fullmatch(r"[\s\[\](){}.,:;*-]*([A-N])[\s\[\](){}.,:;*-]*", fragment)
    if match and match.group(1) in labels:
        return match.group(1)
    return None


def parse_sequence(prediction: str, count: int) -> str | None:
    fragment = answer_fragment(prediction)
    match = re.fullmatch(r"\s*(\d+(?:\s*(?:->|,|-)\s*\d+)+)\s*[.]?\s*", fragment)
    if not match:
        return None
    values = [int(value) for value in re.findall(r"\d+", match.group(1))]
    if len(values) != count or set(values) != set(range(1, count + 1)):
        return None
    return "->".join(map(str, values))


def parse_interval_prediction(prediction: str) -> list[float] | None:
    fragment = answer_fragment(prediction)
    match = re.fullmatch(
        r"\s*[\[(]?\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*[\])]?\s*[.]?\s*",
        fragment,
    )
    if not match:
        return None
    values = [float(match.group(1)), float(match.group(2))]
    return values if values[0] <= values[1] else None


def normalize_answer(task: str, prediction: str, pair: dict[str, Any]) -> Any:
    if task in MULTIPLE_CHOICE_TASKS:
        return parse_choices(
            prediction,
            option_labels(pair["options"]),
            multiple=task == "BU",
        )
    if task == "PSS":
        return parse_sequence(prediction, len(pair["segments"]))
    if task == "FSA":
        return parse_interval_prediction(prediction)
    return prediction.strip()


def response_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            if isinstance(item, dict)
            else getattr(item, "text", "")
            for item in content
        )
    return str(content or "")


def state_path(results_dir: Path, task: str, suffix: str = "run") -> Path:
    return results_dir / ".state" / f"{task}_{suffix}.jsonl"


def manifest_path(state: Path) -> Path:
    return state.with_suffix(".manifest.json")


def ensure_manifest(state: Path, expected: dict[str, Any]) -> None:
    path = manifest_path(state)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit(f"Run configuration does not match {path}")
        return
    if state.exists() and state.stat().st_size:
        raise SystemExit(f"Refusing to resume {state} without its manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
                raise SystemExit(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def latest_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in records}


def selected_pairs(pairs: list[dict[str, Any]], start: int, limit: int | None) -> list[dict[str, Any]]:
    selected = pairs[start:]
    return selected[:limit] if limit is not None else selected


def write_result(path: Path, pairs: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> None:
    result = [
        {key: value for key, value in records[str(pair["id"])].items() if key != "efficiency"}
        for pair in pairs
        if str(pair["id"]) in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def make_client(args: argparse.Namespace) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install dependencies with: uv sync --extra crossvid") from exc
    return OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.retries,
    )


def run_task(task: str, args: argparse.Namespace) -> None:
    model_family = resolve_model_family(
        args.model, getattr(args, "model_family", "auto")
    )
    try:
        extra_body = chat_template_extra_body(model_family, args.enable_thinking)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    qa_path = args.qa_dir / f"{task}.json"
    all_pairs = read_list(qa_path)
    pairs = selected_pairs(all_pairs, args.start, args.limit)
    if not pairs:
        raise SystemExit(f"No selected examples in {qa_path}")

    module = load_official_module(task, args.vendor_root)
    module.model = args.model
    module.total_frames = args.frames
    module.length = args.length
    module.video_root = str(args.uav_root if task in {"MSR", "MOC"} else args.video_root)

    client = make_client(args)
    thread_state = threading.local()

    def chat(messages: list[dict[str, Any]]) -> str:
        try:
            request: dict[str, Any] = {
                "model": args.model,
                "messages": adapt_messages(messages, model_family),
                "temperature": 0,
                "max_tokens": args.max_tokens,
            }
            if extra_body is not None:
                request["extra_body"] = extra_body
            thread_state.api_started = time.perf_counter()
            prediction, efficiency = stream_chat_completion(client, request)
            thread_state.efficiency = efficiency
            return prediction
        except Exception as exc:
            thread_state.error = exc
            if getattr(thread_state, "api_started", None) is not None:
                thread_state.efficiency = {
                    "schema_version": EFFICIENCY_SCHEMA_VERSION,
                    "api_request_ms": 1000
                    * (time.perf_counter() - thread_state.api_started),
                }
            raise

    module.chat = chat
    state = state_path(args.results_dir, task)
    manifest = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "crossvid_revision": CROSSVID_REVISION,
        "task": task,
        "model": args.model,
        "model_family": model_family,
        "base_url": args.base_url,
        "frames": args.frames,
        "length": args.length,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "stream": True,
        "efficiency_schema_version": EFFICIENCY_SCHEMA_VERSION,
        "workers": args.workers,
        "timeout": args.timeout,
        "retries": args.retries,
        "backend_signature": getattr(args, "backend_signature", None),
        "ids": [pair["id"] for pair in pairs],
    }
    ensure_manifest(state, manifest)
    existing = latest_by_id(read_jsonl(state))
    pending = [pair for pair in pairs if not existing.get(str(pair["id"]), {}).get("success")]
    write_lock = threading.Lock()
    completed = len(pairs) - len(pending)

    def infer(pair: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        thread_state.error = None
        thread_state.efficiency = None
        thread_state.api_started = None
        try:
            evaluated = module.evaluate(pair, max_tries=1)
            if evaluated is None:
                error = thread_state.error
                detail = f"{type(error).__name__}: {error}" if error else "official preprocessing failed"
                raise RuntimeError(detail)
            prediction = evaluated[1]
            answer = normalize_answer(task, prediction, pair)
            efficiency = thread_state.efficiency or {
                "schema_version": EFFICIENCY_SCHEMA_VERSION
            }
            if thread_state.api_started is not None:
                efficiency["preprocessing_ms"] = 1000 * (
                    thread_state.api_started - started
                )
            efficiency["end_to_end_ms"] = 1000 * (time.perf_counter() - started)
            return {
                "id": pair["id"],
                "answer": answer,
                "raw_prediction": prediction,
                "success": True,
                "parse_error": answer is None,
                "error": None,
                "model": args.model,
                "efficiency": efficiency,
            }
        except Exception as exc:  # noqa: BLE001 - failures must be persisted per sample.
            efficiency = thread_state.efficiency or {
                "schema_version": EFFICIENCY_SCHEMA_VERSION
            }
            if thread_state.api_started is not None:
                efficiency["preprocessing_ms"] = 1000 * (
                    thread_state.api_started - started
                )
            efficiency["end_to_end_ms"] = 1000 * (time.perf_counter() - started)
            return {
                "id": pair["id"],
                "answer": None,
                "raw_prediction": None,
                "success": False,
                "parse_error": False,
                "error": f"{type(exc).__name__}: {exc}",
                "model": args.model,
                "efficiency": efficiency,
            }

    def persist(record: dict[str, Any]) -> None:
        nonlocal completed
        with write_lock, state.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            existing[str(record["id"])] = record
            completed += 1
            print(
                f"{task} [{completed}/{len(pairs)}] id={record['id']} "
                f"answer={record['answer']} success={record['success']}",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(infer, pair) for pair in pending]
        for future in as_completed(futures):
            persist(future.result())

    result_path = args.results_dir / f"{task}_result.json"
    write_result(result_path, pairs, existing)
    failures = sum(not existing[str(pair["id"])]["success"] for pair in pairs)
    invalid = sum(existing[str(pair["id"])].get("parse_error", False) for pair in pairs)
    print(f"Wrote {result_path} | failures={failures} | unparseable={invalid}")
    efficiency = summarize_efficiency([existing[str(pair["id"])] for pair in pairs])
    print(format_efficiency_summary(efficiency))
    if failures:
        raise SystemExit(f"{task} has {failures} failed examples; rerun to retry them")


def run_benchmarks(args: argparse.Namespace) -> None:
    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        run_task(task, args)


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("judge response did not contain a JSON object")


def judge_ccqa(args: argparse.Namespace) -> None:
    qa_path = args.qa_dir / "CCQA.json"
    result_path = args.results_dir / "CCQA_result.json"
    annotations = index_by_id(read_list(qa_path), qa_path)
    answers = index_by_id(read_list(result_path), result_path)
    if set(annotations) != set(answers):
        raise SystemExit("CCQA judging requires a complete CCQA_result.json")

    client = make_client(args)
    state = state_path(args.results_dir, "CCQA", "judge")
    ids = list(annotations)
    ensure_manifest(
        state,
        {
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "judge_model": args.model,
            "base_url": args.base_url,
            "ids": ids,
        },
    )
    existing = latest_by_id(read_jsonl(state))
    pending = [item_id for item_id in ids if not existing.get(item_id, {}).get("success")]
    write_lock = threading.Lock()
    completed = len(ids) - len(pending)

    def judge(item_id: str) -> dict[str, Any]:
        pair = annotations[item_id]
        answer = answers[item_id].get("answer")
        try:
            prompt = JUDGE_PROMPT.format(
                question=pair["question"],
                answer=pair["answer"],
                points=pair["scoring_points"],
                output=answer,
            )
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=args.max_tokens,
            )
            raw = response_text(response)
            scoring = extract_json_object(raw)
            coverage = scoring.get("coverage")
            correctness = scoring.get("correctness")
            count = len(pair["scoring_points"])
            if (
                not isinstance(coverage, list)
                or not isinstance(correctness, list)
                or len(coverage) != count
                or len(correctness) != count
                or not all(isinstance(value, bool) for value in coverage + correctness)
            ):
                raise ValueError("judge arrays do not match the scoring points")
            return {
                "id": pair["id"],
                "answer": answer,
                "coverage": coverage,
                "correctness": correctness,
                "score": sum(coverage) + sum(correctness),
                "raw_judgment": raw,
                "judge_model": args.model,
                "success": True,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - failures must be persisted per sample.
            return {
                "id": pair["id"],
                "answer": answer,
                "coverage": None,
                "correctness": None,
                "score": 0,
                "raw_judgment": None,
                "judge_model": args.model,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def persist(record: dict[str, Any]) -> None:
        nonlocal completed
        with write_lock, state.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            existing[str(record["id"])] = record
            completed += 1
            print(
                f"CCQA judge [{completed}/{len(ids)}] id={record['id']} "
                f"success={record['success']}",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(judge, item_id) for item_id in pending]
        for future in as_completed(futures):
            persist(future.result())

    output = args.results_dir / "CCQA_score.json"
    ordered_pairs = [annotations[item_id] for item_id in ids]
    write_result(output, ordered_pairs, existing)
    failures = sum(not existing[item_id]["success"] for item_id in ids)
    print(f"Wrote {output} | failures={failures}")
    if failures:
        raise SystemExit(f"CCQA judging has {failures} failed examples; rerun to retry")


def common_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run official CrossVid preprocessing and inference")
    common_api_arguments(run_parser)
    run_parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    run_parser.add_argument("--qa-dir", required=True, type=Path)
    run_parser.add_argument("--video-root", required=True, type=Path)
    run_parser.add_argument("--uav-root", required=True, type=Path)
    run_parser.add_argument("--results-dir", required=True, type=Path)
    run_parser.add_argument("--vendor-root", type=Path, default=Path("vendor/CrossVid"))
    run_parser.add_argument("--frames", type=int, default=128)
    run_parser.add_argument("--length", type=int, default=360)
    run_parser.add_argument("--start", type=int, default=0)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--enable-thinking", action="store_true")
    run_parser.add_argument(
        "--model-family", choices=MODEL_FAMILIES, default="auto"
    )
    run_parser.add_argument("--backend-signature")
    run_parser.set_defaults(function=run_benchmarks)

    judge_parser = subparsers.add_parser("judge", help="judge CCQA with the official rubric")
    common_api_arguments(judge_parser)
    judge_parser.add_argument("--qa-dir", required=True, type=Path)
    judge_parser.add_argument("--results-dir", required=True, type=Path)
    judge_parser.set_defaults(function=judge_ccqa)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be at least 1")
    if getattr(args, "frames", 1) < 2:
        raise SystemExit("--frames must be at least 2")
    if getattr(args, "length", 1) < 1:
        raise SystemExit("--length must be at least 1")
    if getattr(args, "start", 0) < 0:
        raise SystemExit("--start must not be negative")
    if getattr(args, "limit", 0) is not None and getattr(args, "limit", 0) < 0:
        raise SystemExit("--limit must not be negative")
    args.function(args)


if __name__ == "__main__":
    main()
