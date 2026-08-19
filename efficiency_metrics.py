"""Collect and compare serving-efficiency metrics for benchmark runs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

EFFICIENCY_SCHEMA_VERSION = 1


def _content_text(content: Any) -> str:
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


def _usage_value(usage: Any, name: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
        if value is None:
            value = (getattr(usage, "model_extra", None) or {}).get(name)
    return value if isinstance(value, int) else None


def stream_chat_completion(client: Any, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Consume a streamed chat completion and return its text and telemetry."""

    started = time.perf_counter()
    first_token_at = None
    completed_at = None
    usage = None
    finish_reason = None
    parts = []
    stream = client.chat.completions.create(
        **request,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        event_at = time.perf_counter()
        if getattr(choice, "finish_reason", None) is not None:
            finish_reason = choice.finish_reason
            completed_at = event_at
        delta = getattr(choice, "delta", None)
        content = _content_text(getattr(delta, "content", None))
        reasoning = _content_text(getattr(delta, "reasoning_content", None))
        if (content or reasoning) and first_token_at is None:
            first_token_at = event_at
        if content:
            parts.append(content)

    finished = time.perf_counter()
    if finish_reason is None or completed_at is None:
        raise RuntimeError("chat completion stream ended without a finish reason")
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    decode_seconds = (
        completed_at - first_token_at if first_token_at is not None else None
    )
    tpot_ms = None
    output_tokens_per_second = None
    if completion_tokens is not None and completion_tokens > 1 and decode_seconds is not None:
        tpot_ms = 1000 * decode_seconds / (completion_tokens - 1)
        if decode_seconds > 0:
            output_tokens_per_second = (completion_tokens - 1) / decode_seconds

    metrics = {
        "schema_version": EFFICIENCY_SCHEMA_VERSION,
        "ttft_ms": 1000 * (first_token_at - started) if first_token_at else None,
        "api_request_ms": 1000 * (finished - started),
        "e2e_generation_ms": 1000 * (completed_at - started),
        "tpot_ms": tpot_ms,
        "output_tokens_per_second": output_tokens_per_second,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "finish_reason": finish_reason,
    }
    # Custom instrumented servers may expose these fields even though the
    # standard OpenAI usage object does not.
    for name in ("visual_tokens_before", "visual_tokens_after"):
        metrics[name] = _usage_value(usage, name)
    return "".join(parts), metrics


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_efficiency(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [record.get("efficiency") for record in records]
    metrics = [item for item in metrics if isinstance(item, dict)]
    summary: dict[str, Any] = {
        "records": len(records),
        "measured_records": len(metrics),
    }
    for name in (
        "ttft_ms",
        "api_request_ms",
        "e2e_generation_ms",
        "preprocessing_ms",
        "end_to_end_ms",
        "tpot_ms",
        "output_tokens_per_second",
    ):
        values = [item[name] for item in metrics if isinstance(item.get(name), (int, float))]
        summary[name] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
        }
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "visual_tokens_before",
        "visual_tokens_after",
    ):
        values = [item[name] for item in metrics if isinstance(item.get(name), int)]
        summary[name] = {
            "count": len(values),
            "total": sum(values),
            "mean": statistics.fmean(values) if values else None,
        }
    visual_pairs = [
        (item["visual_tokens_before"], item["visual_tokens_after"])
        for item in metrics
        if isinstance(item.get("visual_tokens_before"), int)
        and isinstance(item.get("visual_tokens_after"), int)
    ]
    summary["visual_token_compression"] = {
        "paired_records": len(visual_pairs),
        **compression_values(
            sum(before for before, _ in visual_pairs),
            sum(after for _, after in visual_pairs),
        ),
    }
    return summary


def format_efficiency_summary(summary: dict[str, Any]) -> str:
    prompt = summary["prompt_tokens"]["mean"]
    ttft = summary["ttft_ms"]["p50"]
    e2e = summary["end_to_end_ms"]["p50"]
    return (
        f"Efficiency: measured={summary['measured_records']}/{summary['records']} | "
        f"prompt tokens/row={prompt:.1f} | "
        if prompt is not None
        else f"Efficiency: measured={summary['measured_records']}/{summary['records']} | "
    ) + (
        f"TTFT p50={ttft:.2f} ms | " if ttft is not None else "TTFT p50=n/a | "
    ) + (f"end-to-end p50={e2e:.2f} ms" if e2e is not None else "end-to-end p50=n/a")


def compression_values(before: int, after: int) -> dict[str, float | int | None]:
    return {
        "tokens_before": before,
        "tokens_after": after,
        "tokens_removed": before - after,
        "retention_ratio": after / before if before else None,
        "reduction_fraction": 1 - after / before if before else None,
        "compression_factor": before / after if after else None,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_efficiency_records(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    if path.is_file():
        records = _read_jsonl(path)
        return "mmiu", {str(record["index"]): record for record in records}
    mmiu = path / "results.jsonl"
    if mmiu.is_file():
        return run_efficiency_records(mmiu)
    state = path / ".state"
    files = sorted(state.glob("*_run.jsonl")) if state.is_dir() else []
    if files:
        records = {}
        for file in files:
            task = file.name.removesuffix("_run.jsonl")
            for record in _read_jsonl(file):
                records[f"{task}:{record['id']}"] = record
        return "crossvid", records
    raise SystemExit(f"No MMIU or CrossVid inference state found at {path}")


def compare_runs(baseline: Path, candidate: Path) -> dict[str, Any]:
    baseline_name, baseline_records = run_efficiency_records(baseline)
    candidate_name, candidate_records = run_efficiency_records(candidate)
    if baseline_name != candidate_name:
        raise SystemExit("Baseline and candidate use different benchmarks")

    common = sorted(set(baseline_records) & set(candidate_records))
    eligible = [
        key
        for key in common
        if baseline_records[key].get("success") is not False
        and candidate_records[key].get("success") is not False
    ]
    visual_pairs = []
    prompt_pairs = []
    for key in eligible:
        base_metrics = baseline_records[key].get("efficiency", {})
        candidate_metrics = candidate_records[key].get("efficiency", {})
        visual_before = candidate_metrics.get("visual_tokens_before")
        visual_after = candidate_metrics.get("visual_tokens_after")
        if isinstance(visual_before, int) and isinstance(visual_after, int):
            visual_pairs.append((visual_before, visual_after))
        prompt_before = base_metrics.get("prompt_tokens")
        prompt_after = candidate_metrics.get("prompt_tokens")
        if isinstance(prompt_before, int) and isinstance(prompt_after, int):
            prompt_pairs.append((prompt_before, prompt_after))
    use_visual = bool(eligible) and len(visual_pairs) == len(eligible)
    pairs = visual_pairs if use_visual else prompt_pairs
    if not pairs:
        raise SystemExit("Runs have no paired examples with visual- or prompt-token usage")

    before = sum(item[0] for item in pairs)
    after = sum(item[1] for item in pairs)
    return {
        "benchmark": baseline_name,
        "common_examples": len(common),
        "paired_examples": len(pairs),
        "baseline_only": len(set(baseline_records) - set(candidate_records)),
        "candidate_only": len(set(candidate_records) - set(baseline_records)),
        "metric": (
            "paired_visual_tokens" if use_visual else "paired_total_prompt_tokens"
        ),
        **compression_values(before, after),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    comparison = compare_runs(args.baseline, args.candidate)
    if args.json:
        print(json.dumps(comparison, indent=2))
        return
    print(f"Metric: {comparison['metric']}")
    print(
        f"Paired examples: {comparison['paired_examples']}/{comparison['common_examples']} | "
        f"baseline-only={comparison['baseline_only']} | candidate-only={comparison['candidate_only']}"
    )
    print(
        f"Tokens: {comparison['tokens_before']} -> {comparison['tokens_after']} "
        f"({comparison['tokens_removed']} removed)"
    )
    retention = comparison["retention_ratio"]
    reduction = comparison["reduction_fraction"]
    factor = comparison["compression_factor"]
    print(
        f"Retention: {retention:.4f} | " if retention is not None else "Retention: n/a | ",
        f"Reduction: {reduction:.4f} | " if reduction is not None else "Reduction: n/a | ",
        f"Compression factor: {factor:.4f}x" if factor is not None else "Compression factor: n/a",
        sep="",
    )


if __name__ == "__main__":
    main()
