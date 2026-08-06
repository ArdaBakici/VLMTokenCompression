"""Summarize every completed benchmark run under a results directory.

Scores each run the way its own scorer does and prints one row per run, so that
models, pruning rates and server profiles can be compared at a glance. MMIU runs
are rescored from their JSONL; CrossVid runs report the aggregate that
`crossvid-score score` wrote, since recomputing it needs the annotations.

The columns that are not accuracy matter as much as the one that is. A run whose
`fail` or `invalid` count differs from its neighbour is not comparable to it: API
failures and unparseable answers are both scored as incorrect, so either can move
the headline number without the model behaving differently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmiu_eval import (
    latest_by_index,
    manifest_path,
    read_jsonl,
    reparse_records,
    score_records,
)

MMIU_RESULT = "results.jsonl"
CROSSVID_SUMMARY = "summary.json"
SERVER_CONFIG = "server-config.json"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def profile(run: Path, manifest: dict[str, Any]) -> str:
    """Describe the server profile a run was produced with."""

    config = read_json(run / SERVER_CONFIG)
    if not config:
        return "baseline" if manifest else "unknown"
    parts = [f"prune={config.get('image_pruning_rate', '?')}"]
    layer = config.get("vit_attention_score_layer_index")
    if layer is not None:
        parts.append(f"layer={layer}")
    if config.get("glibc_shim", "none") != "none":
        parts.append("glibc-shim")
    return " ".join(parts)


def mmiu_summary(results: Path, dataset: Any) -> dict[str, Any]:
    records = read_jsonl(results)
    manifest = read_json(manifest_path(results))
    missing = unexpected = 0
    if manifest.get("indices") is not None:
        expected = set(manifest["indices"])
        latest = latest_by_index(records)
        missing = len(expected - set(latest))
        unexpected = len(set(latest) - expected)
        records = [record for index, record in latest.items() if index in expected]

    stored = score_records(records)
    reparsed = None
    if dataset is not None:
        reparsed = score_records(reparse_records(records, dataset))

    score = reparsed or stored
    return {
        "benchmark": "mmiu",
        "run": results.parent,
        "model": manifest.get("model", "unknown"),
        "profile": profile(results.parent, manifest),
        "rows": score["unique_records"],
        "missing": missing,
        "unexpected": unexpected,
        "failures": score["failures"],
        "invalid": score["invalid_predictions"],
        "score": score["macro_accuracy"] * 100,
        "recorded_score": stored["macro_accuracy"] * 100 if reparsed else None,
        "metric": "macro",
    }


def crossvid_model(run: Path) -> str:
    """CrossVid records the model per task in its resume state, not centrally."""

    config = read_json(run / SERVER_CONFIG)
    if config.get("model"):
        return config["model"]
    for manifest in sorted((run / ".state").glob("*_run.manifest.json")):
        model = read_json(manifest).get("model")
        if model:
            return model
    return "unknown"


def crossvid_summary(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    counts = summary.get("counts", {})
    return {
        "benchmark": "crossvid",
        "run": summary_path.parent,
        "model": crossvid_model(summary_path.parent),
        "profile": "baseline",
        "rows": sum(counts.values()),
        "missing": 0,
        "unexpected": 0,
        "failures": 0,
        "invalid": 0,
        "score": summary.get("averages", {}).get("O.Avg", 0.0) * 100,
        "recorded_score": None,
        "metric": "O.Avg",
    }


def collect(root: Path, dataset: Any) -> list[dict[str, Any]]:
    if root.is_file():
        return [mmiu_summary(root, dataset)]
    if not root.is_dir():
        raise SystemExit(f"No such results path: {root}")

    summaries = [
        mmiu_summary(results, dataset) for results in sorted(root.rglob(MMIU_RESULT))
    ]
    summaries += [
        crossvid_summary(summary) for summary in sorted(root.rglob(CROSSVID_SUMMARY))
    ]
    if not summaries:
        raise SystemExit(f"No completed runs found under {root}")
    return sorted(summaries, key=lambda row: (row["benchmark"], row["model"], row["run"]))


def cell(text: str, width: int) -> str:
    if len(text) > width:
        return f"{text[: width - 1]}…"
    return f"{text:<{width}}"


def print_table(summaries: list[dict[str, Any]], root: Path) -> None:
    columns = (
        ("benchmark", "benchmark", 9),
        ("model", "model", 30),
        ("profile", "profile", 30),
        ("rows", "rows", 6),
        ("failures", "fail", 5),
        ("invalid", "invalid", 13),
        ("score", "score", 7),
    )
    print("  ".join(f"{title:<{width}}" for _, title, width in columns))
    print("  ".join("-" * width for _, _, width in columns))
    for row in summaries:
        cells = []
        for key, _, width in columns:
            value = row[key]
            if key == "score":
                text = f"{value:.2f}"
            elif key == "invalid":
                share = 100 * value / row["rows"] if row["rows"] else 0.0
                text = f"{value} ({share:.1f}%)"
            else:
                text = str(value)
            cells.append(cell(text, width))
        print("  ".join(cells))

    print()
    for row in summaries:
        try:
            location = row["run"].relative_to(root if root.is_dir() else root.parent)
        except ValueError:
            location = row["run"]
        note = f"{row['metric']} {row['score']:.2f}"
        recorded = row["recorded_score"]
        if recorded is not None and abs(recorded - row["score"]) >= 0.005:
            note += f" (recorded {recorded:.2f} before re-parsing)"
        flags = []
        if row["missing"] or row["unexpected"]:
            flags.append(f"{row['missing']} missing, {row['unexpected']} unexpected rows")
        if row["failures"]:
            flags.append(f"{row['failures']} API failures counted as incorrect")
        if flags:
            note += " | " + "; ".join(flags)
        print(f"{location}: {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        nargs="?",
        default=Path("results"),
        type=Path,
        help="A results tree, a run directory, or a single results.jsonl.",
    )
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="Rescore MMIU predictions with the current extractor, without "
        "modifying any results file.",
    )
    parser.add_argument("--dataset-path", help="optional local all.parquet path")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead.")
    args = parser.parse_args()

    dataset = None
    if args.reparse:
        from mmiu_eval import load_mmiu

        dataset = load_mmiu(args.dataset_path)

    summaries = collect(args.results, dataset)
    if args.json:
        print(
            json.dumps(
                [{**row, "run": str(row["run"])} for row in summaries], indent=2
            )
        )
        return
    print_table(summaries, args.results)


if __name__ == "__main__":
    main()
