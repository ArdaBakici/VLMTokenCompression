import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.show_invalid import classify, find_results, report


def write_run(directory: Path, records: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    results = directory / "results.jsonl"
    with results.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (directory / "results.jsonl.manifest.json").write_text(
        json.dumps({"model": "test-model", "max_tokens": 16, "enable_thinking": False}),
        encoding="utf-8",
    )
    return results


def record(index: int, **overrides) -> dict:
    base = {
        "index": index,
        "task": "video_captioning",
        "ground_truth": "B",
        "success": True,
        "prediction": "B",
        "choice": "B",
        "error": None,
        "num_images": 12,
    }
    return {**base, **overrides}


class ClassifyTest(unittest.TestCase):
    def test_names_each_reason(self):
        self.assertEqual(classify(""), "empty prediction")
        self.assertEqual(classify(None), "empty prediction")
        self.assertEqual(
            classify("The answer is F"), "read as 'F', which the row did not offer"
        )
        self.assertEqual(
            classify("Frames 1 and 2 differ, so I would say the"),
            "letter present, but not where the answer is read from",
        )
        self.assertEqual(
            classify("Based on the sequence shown across the two videos, the"),
            "no standalone option letter",
        )


class ReportTest(unittest.TestCase):
    def test_counts_only_unparsed_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            results = write_run(
                Path(directory) / "run",
                [
                    record(0),
                    record(1, choice=None, prediction="the"),
                    record(2, success=False, prediction=None, choice=None, error="X: y"),
                    # An unlabeled row is skipped by scoring and by this report.
                    record(3, ground_truth=None, choice=None, prediction="the"),
                ],
            )
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(report(results, limit=20, task=None, width=200), 1)
            printed = output.getvalue()
            self.assertIn("api_failures=1", printed)
            self.assertIn("max_tokens=16", printed)
            self.assertIn("index=1", printed)

    def test_keeps_only_the_last_record_per_index(self):
        with tempfile.TemporaryDirectory() as directory:
            results = write_run(
                Path(directory) / "run",
                [record(0, choice=None, prediction="the"), record(0)],
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(report(results, limit=20, task=None, width=200), 0)

    def test_filters_by_task(self):
        with tempfile.TemporaryDirectory() as directory:
            results = write_run(
                Path(directory) / "run",
                [
                    record(0, choice=None, prediction="the"),
                    record(1, task="temporal_ordering", choice=None, prediction="the"),
                ],
            )
            with redirect_stdout(io.StringIO()):
                count = report(results, limit=20, task="temporal_ordering", width=200)
            self.assertEqual(count, 1)


class FindResultsTest(unittest.TestCase):
    def test_accepts_a_file_a_run_and_a_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_run(root / "model-a" / "run-1", [record(0)])
            second = write_run(root / "model-b" / "run-2", [record(0)])

            self.assertEqual(find_results(first), [first])
            self.assertEqual(find_results(first.parent), [first])
            self.assertEqual(find_results(root), [first, second])

    def test_rejects_a_path_without_results(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                find_results(Path(directory))
            with self.assertRaises(SystemExit):
                find_results(Path(directory) / "absent")


if __name__ == "__main__":
    unittest.main()
