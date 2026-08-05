"""Show the MMIU predictions that strict scoring discards.

Strict scoring counts two kinds of discarded row as incorrect: an API failure,
which stores an error and no prediction, and an unparseable prediction, which
answered but never yielded an option letter. This report separates them, groups
them by task, and prints the raw text so that the cause is visible.

The stored records do not include the option list or the completion's finish
reason, so the classification of an unparseable prediction is a triage hint
rather than a verdict.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mmiu_eval import latest_by_index, manifest_path, parse_choice, read_jsonl

RESULT_NAME = "results.jsonl"
# Option letters MMIU uses. The stored record does not say which subset a row
# offered, so a letter found here only suggests that the answer is recoverable.
ANY_LABEL = set("ABCDEFGHIJKLMN")
# A bare capital letter anywhere in the text. "I" and "A" also start ordinary
# sentences, so this over-reports slightly by design.
STANDALONE_LABEL = re.compile(r"\b[A-N]\b")


def find_results(path: Path) -> list[Path]:
    """Accept a JSONL file, a run directory, or a tree of run directories."""

    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"No such results path: {path}")
    found = sorted(path.rglob(RESULT_NAME))
    if not found:
        raise SystemExit(f"No {RESULT_NAME} found under {path}")
    return found


def classify(prediction: str | None) -> str:
    """Describe why a successful call produced no option letter."""

    text = (prediction or "").strip()
    if not text:
        return "empty prediction"
    letter = parse_choice(text, ANY_LABEL)
    if letter is not None:
        # It parses against every MMIU label but did not parse during the run,
        # so the row offered fewer options than this letter. Naming the letter
        # keeps the prose cases visible: "I cannot tell" reads as 'I' here.
        return f"read as {letter!r}, which the row did not offer"
    if STANDALONE_LABEL.search(text):
        return "letter present, but not where the answer is read from"
    return "no standalone option letter"


def manifest_settings(results: Path) -> dict[str, Any]:
    path = manifest_path(results)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def report(results: Path, limit: int, task: str | None, width: int) -> int:
    records = latest_by_index(read_jsonl(results))
    failed = [r for r in records.values() if not r.get("success")]
    invalid = [
        r
        for r in records.values()
        if r.get("success") and r.get("ground_truth") is not None and r.get("choice") is None
    ]
    if task is not None:
        failed = [r for r in failed if r["task"] == task]
        invalid = [r for r in invalid if r["task"] == task]

    settings = manifest_settings(results)
    print(f"\n{results}")
    if settings:
        print(
            f"  model={settings.get('model')} max_tokens={settings.get('max_tokens')} "
            f"enable_thinking={settings.get('enable_thinking')}"
        )
    share = 100 * len(invalid) / len(records) if records else 0.0
    print(
        f"  records={len(records)} api_failures={len(failed)} "
        f"invalid={len(invalid)} ({share:.2f}%)"
    )

    if invalid:
        print("\n  invalid predictions per task")
        for name, count in Counter(r["task"] for r in invalid).most_common():
            print(f"    {count:5d}  {name}")

        print("\n  why they did not parse")
        for reason, count in Counter(classify(r["prediction"]) for r in invalid).most_common():
            print(f"    {count:5d}  {reason}")

        lengths = sorted(len((r["prediction"] or "").strip()) for r in invalid)
        median = lengths[len(lengths) // 2]
        print(
            f"\n  prediction length in characters: min={lengths[0]} "
            f"median={median} max={lengths[-1]}"
        )
        if settings.get("max_tokens"):
            print(
                f"  compare against max_tokens={settings['max_tokens']}; predictions "
                "that all stop near one length were cut off, not malformed"
            )

        shown = invalid if limit == 0 else invalid[:limit]
        print(f"\n  {len(shown)} of {len(invalid)} invalid rows")
        for record in shown:
            text = (record["prediction"] or "").strip().replace("\n", "\\n")
            if width and len(text) > width:
                text = f"{text[:width]}..."
            print(
                f"    index={record['index']} task={record['task']} "
                f"truth={record.get('ground_truth')!r} images={record.get('num_images')}"
            )
            print(f"      {text!r}")

    if failed:
        print("\n  api failures per error")
        for error, count in Counter(
            str(r.get("error")).split(":")[0] for r in failed
        ).most_common():
            print(f"    {count:5d}  {error}")

    return len(invalid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        help=f"a {RESULT_NAME} file, a run directory, or a tree of run directories",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="invalid rows to print per result file; 0 prints all",
    )
    parser.add_argument("--task", help="restrict the report to one MMIU task")
    parser.add_argument(
        "--width",
        type=int,
        default=200,
        help="characters of each prediction to print; 0 prints all",
    )
    args = parser.parse_args()

    total = sum(
        report(results, args.limit, args.task, args.width)
        for results in find_results(args.results)
    )
    print(f"\nTotal invalid predictions: {total}")


if __name__ == "__main__":
    main()
