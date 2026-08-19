import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from efficiency_metrics import (
    compare_runs,
    compression_values,
    stream_chat_completion,
    summarize_efficiency,
)


class FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.request = None

    def create(self, **request):
        self.request = request
        return iter(self.chunks)


class StreamingTest(unittest.TestCase):
    def test_collects_text_usage_and_latency(self):
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason=None)],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="B"), finish_reason=None)],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="."), finish_reason="stop")],
                usage=None,
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=2,
                    total_tokens=102,
                ),
            ),
        ]
        completions = FakeCompletions(chunks)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        text, metrics = stream_chat_completion(client, {"model": "test"})

        self.assertEqual(text, "B.")
        self.assertTrue(completions.request["stream"])
        self.assertEqual(
            completions.request["stream_options"], {"include_usage": True}
        )
        self.assertEqual(metrics["prompt_tokens"], 100)
        self.assertEqual(metrics["completion_tokens"], 2)
        self.assertIsNotNone(metrics["ttft_ms"])
        self.assertIsNotNone(metrics["tpot_ms"])

    def test_rejects_a_stream_without_terminal_completion(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="partial"), finish_reason=None
                    )
                ],
                usage=None,
            )
        ]
        completions = FakeCompletions(chunks)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with self.assertRaisesRegex(RuntimeError, "without a finish reason"):
            stream_chat_completion(client, {"model": "test"})


class SummaryTest(unittest.TestCase):
    def test_summarizes_latency_tokens_and_measured_visual_compression(self):
        records = [
            {
                "efficiency": {
                    "ttft_ms": 10.0,
                    "end_to_end_ms": 20.0,
                    "preprocessing_ms": 5.0,
                    "prompt_tokens": 100,
                    "completion_tokens": 2,
                    "total_tokens": 102,
                    "visual_tokens_before": 80,
                    "visual_tokens_after": 40,
                }
            },
            {
                "efficiency": {
                    "ttft_ms": 30.0,
                    "end_to_end_ms": 40.0,
                    "preprocessing_ms": 15.0,
                    "prompt_tokens": 200,
                    "completion_tokens": 2,
                    "total_tokens": 202,
                    "visual_tokens_before": 120,
                    "visual_tokens_after": 60,
                }
            },
        ]
        summary = summarize_efficiency(records)
        self.assertEqual(summary["ttft_ms"]["p50"], 20.0)
        self.assertEqual(summary["preprocessing_ms"]["p50"], 10.0)
        self.assertEqual(summary["prompt_tokens"]["mean"], 150.0)
        self.assertEqual(
            summary["visual_token_compression"],
            {"paired_records": 2, **compression_values(200, 100)},
        )


class ComparisonTest(unittest.TestCase):
    def test_compares_prompt_tokens_only_for_paired_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            baseline.write_text(
                "\n".join(
                    json.dumps(
                        {"index": index, "efficiency": {"prompt_tokens": tokens}}
                    )
                    for index, tokens in ((0, 100), (1, 200), (2, 300))
                )
                + "\n",
                encoding="utf-8",
            )
            candidate.write_text(
                "\n".join(
                    json.dumps(
                        {"index": index, "efficiency": {"prompt_tokens": tokens}}
                    )
                    for index, tokens in ((0, 50), (1, 100))
                )
                + "\n",
                encoding="utf-8",
            )

            comparison = compare_runs(baseline, candidate)

            self.assertEqual(comparison["paired_examples"], 2)
            self.assertEqual(comparison["tokens_before"], 300)
            self.assertEqual(comparison["tokens_after"], 150)
            self.assertEqual(comparison["reduction_fraction"], 0.5)
            self.assertEqual(comparison["compression_factor"], 2.0)

    def test_prefers_measured_visual_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            baseline.write_text(
                json.dumps({"index": 0, "efficiency": {"prompt_tokens": 100}})
                + "\n",
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(
                    {
                        "index": 0,
                        "efficiency": {
                            "prompt_tokens": 80,
                            "visual_tokens_before": 60,
                            "visual_tokens_after": 30,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            comparison = compare_runs(baseline, candidate)

            self.assertEqual(comparison["metric"], "paired_visual_tokens")
            self.assertEqual(comparison["tokens_before"], 60)
            self.assertEqual(comparison["tokens_after"], 30)

    def test_partial_visual_coverage_keeps_the_prompt_token_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            baseline.write_text(
                "\n".join(
                    json.dumps(
                        {"index": index, "efficiency": {"prompt_tokens": 100}}
                    )
                    for index in (0, 1)
                )
                + "\n",
                encoding="utf-8",
            )
            candidate.write_text(
                "\n".join(
                    (
                        json.dumps(
                            {
                                "index": 0,
                                "efficiency": {
                                    "prompt_tokens": 50,
                                    "visual_tokens_before": 80,
                                    "visual_tokens_after": 20,
                                },
                            }
                        ),
                        json.dumps(
                            {"index": 1, "efficiency": {"prompt_tokens": 50}}
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            comparison = compare_runs(baseline, candidate)

            self.assertEqual(comparison["metric"], "paired_total_prompt_tokens")
            self.assertEqual(comparison["paired_examples"], 2)


if __name__ == "__main__":
    unittest.main()
