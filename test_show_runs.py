import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.show_runs import cell, collect, crossvid_model, mmiu_summary, profile

DATASET = (
    {"task": "video_captioning", "options": "A: yes\nB: no"},
    {"task": "video_captioning", "options": "A: yes\nB: no"},
)
PREDICTIONS = ("B. no", "A. yes")
TRUTHS = ("B", "A")


def write_mmiu_run(run: Path, model: str, choices, config=None) -> Path:
    run.mkdir(parents=True)
    results = run / "results.jsonl"
    with results.open("w", encoding="utf-8") as handle:
        for index, (prediction, truth, choice) in enumerate(
            zip(PREDICTIONS, TRUTHS, choices)
        ):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "task": DATASET[index]["task"],
                        "ground_truth": truth,
                        "model": model,
                        "num_images": 2,
                        "success": True,
                        "prediction": prediction,
                        "choice": choice,
                        "error": None,
                    }
                )
                + "\n"
            )
    (run / "results.jsonl.manifest.json").write_text(
        json.dumps({"model": model, "indices": [0, 1]}), encoding="utf-8"
    )
    if config is not None:
        (run / "server-config.json").write_text(json.dumps(config), encoding="utf-8")
    return results


class CellTest(unittest.TestCase):
    def test_pads_and_truncates(self):
        self.assertEqual(cell("ab", 4), "ab  ")
        self.assertEqual(cell("abcdef", 4), "abc…")


class ProfileTest(unittest.TestCase):
    def test_names_the_compression_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "server-config.json").write_text(
                json.dumps(
                    {
                        "image_pruning_rate": "0.3",
                        "vit_attention_score_layer_index": "-2",
                        "glibc_shim": "glibc-log2-downgrade-v1",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(profile(run, {"model": "m"}), "prune=0.3 layer=-2 glibc-shim")

    def test_calls_a_run_without_a_server_config_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(profile(Path(directory), {"model": "m"}), "baseline")
            self.assertEqual(profile(Path(directory), {}), "unknown")


class MmiuSummaryTest(unittest.TestCase):
    def test_counts_unparsed_answers_and_reparses(self):
        with tempfile.TemporaryDirectory() as directory:
            results = write_mmiu_run(Path(directory) / "run", "m", [None, None])

            stored = mmiu_summary(results, dataset=None)
            self.assertEqual(stored["invalid"], 2)
            self.assertEqual(stored["score"], 0.0)
            self.assertIsNone(stored["recorded_score"])

            reparsed = mmiu_summary(results, dataset=DATASET)
            self.assertEqual(reparsed["invalid"], 0)
            self.assertEqual(reparsed["score"], 100.0)
            self.assertEqual(reparsed["recorded_score"], 0.0)

    def test_reports_rows_missing_against_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            results = write_mmiu_run(run, "m", ["B", "A"])
            manifest = run / "results.jsonl.manifest.json"
            manifest.write_text(
                json.dumps({"model": "m", "indices": [0, 1, 2]}), encoding="utf-8"
            )
            self.assertEqual(mmiu_summary(results, dataset=None)["missing"], 1)


class CrossvidModelTest(unittest.TestCase):
    def test_falls_back_to_the_per_task_run_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / ".state").mkdir()
            self.assertEqual(crossvid_model(run), "unknown")
            (run / ".state" / "BU_run.manifest.json").write_text(
                json.dumps({"model": "Qwen/Qwen3-VL-8B-Instruct"}), encoding="utf-8"
            )
            self.assertEqual(crossvid_model(run), "Qwen/Qwen3-VL-8B-Instruct")


class CollectTest(unittest.TestCase):
    def test_finds_both_benchmarks_and_sorts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_mmiu_run(root / "b-mmiu" / "run", "model-b", ["B", "A"])
            write_mmiu_run(root / "a-mmiu" / "run", "model-a", ["B", "A"])
            crossvid = root / "a-crossvid" / "run"
            (crossvid / ".state").mkdir(parents=True)
            (crossvid / "summary.json").write_text(
                json.dumps({"counts": {"BU": 5}, "averages": {"O.Avg": 0.5}}),
                encoding="utf-8",
            )

            summaries = collect(root, dataset=None)
            self.assertEqual(
                [(row["benchmark"], row["model"]) for row in summaries],
                [
                    ("crossvid", "unknown"),
                    ("mmiu", "model-a"),
                    ("mmiu", "model-b"),
                ],
            )
            self.assertEqual(summaries[0]["score"], 50.0)

    def test_rejects_a_tree_without_runs(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit):
            collect(Path(directory), dataset=None)

    def test_accepts_a_single_results_file(self):
        with tempfile.TemporaryDirectory() as directory:
            results = write_mmiu_run(Path(directory) / "run", "m", ["B", "A"])
            with redirect_stdout(io.StringIO()):
                summaries = collect(results, dataset=None)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["score"], 100.0)


if __name__ == "__main__":
    unittest.main()
