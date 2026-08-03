import json
import tempfile
import unittest
from pathlib import Path

from crossvid_score import (
    TASKS,
    evaluate,
    interval_iou,
    parse_interval,
    score_ccqa,
    score_exact,
)


class IntervalTest(unittest.TestCase):
    def test_iou_matches_official_definition(self):
        self.assertEqual(interval_iou((1, 3), (2, 4)), 1 / 3)
        self.assertEqual(interval_iou((1, 2), (3, 4)), 0)

    def test_interval_parser_accepts_saved_and_raw_forms(self):
        self.assertEqual(parse_interval("1,2.5"), (1.0, 2.5))
        self.assertEqual(parse_interval([1, 2.5]), (1.0, 2.5))
        self.assertIsNone(parse_interval("not an interval"))


class ExactScoreTest(unittest.TestCase):
    def test_bu_joins_multiple_reference_choices(self):
        annotations = {"0": {"id": 0, "answer": ["B", "C"]}}
        results = {"0": {"id": 0, "answer": "BC"}}
        self.assertEqual(score_exact("BU", annotations, results), 1)


class CcqaScoreTest(unittest.TestCase):
    def test_coverage_and_correctness_have_equal_weight(self):
        annotations = {"0": {"id": 0, "scoring_points": ["one", "two"]}}
        results = {
            "0": {
                "id": 0,
                "coverage": [True, True],
                "correctness": [True, False],
            }
        }
        self.assertEqual(score_ccqa(annotations, results), 0.75)


class EndToEndTest(unittest.TestCase):
    def test_perfect_complete_run_scores_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qa_dir = root / "QA"
            results_dir = root / "results"
            qa_dir.mkdir()
            results_dir.mkdir()
            for task in TASKS:
                pair = {"id": 0, "answer": ["A"] if task == "BU" else "A"}
                result = {"id": 0, "answer": "A"}
                result_name = f"{task}_result.json"
                if task == "FSA":
                    pair["answer"] = [1, 3]
                    result["answer"] = [1, 3]
                elif task == "CCQA":
                    pair["scoring_points"] = ["point"]
                    result = {
                        "id": 0,
                        "coverage": [True],
                        "correctness": [True],
                    }
                    result_name = "CCQA_score.json"
                (qa_dir / f"{task}.json").write_text(json.dumps([pair]))
                (results_dir / result_name).write_text(json.dumps([result]))

            summary = evaluate(qa_dir, results_dir)
            self.assertTrue(all(score == 1 for score in summary["scores"].values()))
            self.assertEqual(summary["averages"]["O.Avg"], 1)


if __name__ == "__main__":
    unittest.main()
