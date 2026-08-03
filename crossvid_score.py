"""Strictly validate and score outputs from the official CrossVid scripts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TASKS = ("BU", "NC", "CC", "PEA", "PI", "FSA", "PSS", "MSR", "MOC", "CCQA")
EXACT_TASKS = set(TASKS) - {"FSA", "CCQA"}
DIMENSIONS = {
    "C.Avg": ("BU", "NC", "CC", "PEA"),
    "T.Avg": ("PI", "FSA", "PSS"),
    "M.Avg": ("MSR", "MOC"),
}


def read_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, list):
        raise SystemExit(f"Expected a JSON list in {path}")
    return value


def index_by_id(items: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed = {}
    for item in items:
        if "id" not in item:
            raise SystemExit(f"Record without an id in {path}")
        item_id = str(item["id"])
        if item_id in indexed:
            raise SystemExit(f"Duplicate id {item['id']!r} in {path}")
        indexed[item_id] = item
    return indexed


def require_complete(
    annotations: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    result_path: Path,
) -> None:
    expected = set(annotations)
    actual = set(results)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"{len(missing)} missing")
        if extra:
            details.append(f"{len(extra)} unknown")
        raise SystemExit(f"Incomplete {result_path}: {', '.join(details)} IDs")


def expected_answer(task: str, pair: dict[str, Any]) -> Any:
    answer = pair["answer"]
    if task == "BU" and isinstance(answer, list):
        return "".join(answer)
    return answer


def score_exact(
    task: str,
    annotations: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> float:
    correct = sum(
        results[item_id].get("answer") == expected_answer(task, pair)
        for item_id, pair in annotations.items()
    )
    return correct / len(annotations)


def interval_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    start1, end1 = first
    start2, end2 = second
    intersection = max(0.0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union else 0.0


def parse_interval(answer: Any) -> tuple[float, float] | None:
    try:
        if isinstance(answer, str):
            values = answer.split(",")
        elif isinstance(answer, list | tuple):
            values = answer
        else:
            return None
        if len(values) != 2:
            return None
        interval = (float(values[0]), float(values[1]))
        if not all(math.isfinite(value) for value in interval):
            return None
        return interval
    except (TypeError, ValueError):
        return None


def score_fsa(
    annotations: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]]
) -> float:
    total = 0.0
    for item_id, pair in annotations.items():
        predicted = parse_interval(results[item_id].get("answer"))
        reference = parse_interval(pair["answer"])
        if reference is None:
            raise SystemExit(f"Invalid FSA reference interval for id {pair['id']!r}")
        if predicted is not None:
            total += interval_iou(predicted, reference)
    return total / len(annotations)


def score_ccqa(
    annotations: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]]
) -> float:
    earned = 0
    possible = 0
    for item_id, pair in annotations.items():
        result = results[item_id]
        coverage = result.get("coverage")
        correctness = result.get("correctness")
        point_count = len(pair["scoring_points"])
        if not isinstance(coverage, list) or not isinstance(correctness, list):
            raise SystemExit(f"Invalid CCQA judge arrays for id {pair['id']!r}")
        if len(coverage) != point_count or len(correctness) != point_count:
            raise SystemExit(f"Wrong CCQA judge array length for id {pair['id']!r}")
        if not all(isinstance(value, bool) for value in coverage + correctness):
            raise SystemExit(f"Non-boolean CCQA judge value for id {pair['id']!r}")
        earned += sum(coverage) + sum(correctness)
        possible += 2 * point_count
    return earned / possible


def evaluate(qa_dir: Path, results_dir: Path) -> dict[str, Any]:
    scores = {}
    counts = {}
    for task in TASKS:
        qa_path = qa_dir / f"{task}.json"
        result_name = f"{task}_score.json" if task == "CCQA" else f"{task}_result.json"
        result_path = results_dir / result_name
        annotations = index_by_id(read_list(qa_path), qa_path)
        results = index_by_id(read_list(result_path), result_path)
        if not annotations:
            raise SystemExit(f"No annotations in {qa_path}")
        require_complete(annotations, results, result_path)
        counts[task] = len(annotations)
        if task in EXACT_TASKS:
            scores[task] = score_exact(task, annotations, results)
        elif task == "FSA":
            scores[task] = score_fsa(annotations, results)
        else:
            scores[task] = score_ccqa(annotations, results)

    averages = {
        name: sum(scores[task] for task in tasks) / len(tasks)
        for name, tasks in DIMENSIONS.items()
    }
    averages["O.Avg"] = sum(scores.values()) / len(TASKS)
    return {"counts": counts, "scores": scores, "averages": averages}


def print_summary(summary: dict[str, Any]) -> None:
    print("task,rows,score")
    for task in TASKS:
        print(f"{task},{summary['counts'][task]},{summary['scores'][task] * 100:.4f}")
    for name in ("C.Avg", "T.Avg", "M.Avg", "O.Avg"):
        print(f"{name}: {summary['averages'][name] * 100:.4f}")


def inspect(qa_dir: Path) -> None:
    print("task,rows")
    total = 0
    for task in TASKS:
        path = qa_dir / f"{task}.json"
        count = len(index_by_id(read_list(path), path))
        total += count
        print(f"{task},{count}")
    print(f"Total: {total}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--qa-dir", required=True, type=Path)
    inspect_parser.set_defaults(function=lambda args: inspect(args.qa_dir))

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--qa-dir", required=True, type=Path)
    score_parser.add_argument("--results-dir", required=True, type=Path)
    score_parser.add_argument("--json-output", type=Path)
    score_parser.set_defaults(function=run_score)
    return root


def run_score(args: argparse.Namespace) -> None:
    summary = evaluate(args.qa_dir, args.results_dir)
    print_summary(summary)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
