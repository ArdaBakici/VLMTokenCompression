import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mmiu_eval import (
    build_prompt,
    ensure_manifest,
    option_labels,
    parse_choice,
    reparse_records,
    score_records,
    validate_model_coverage,
)


class ChoiceParsingTest(unittest.TestCase):
    def test_finds_option_labels(self):
        self.assertEqual(option_labels("A: one\nB: two\nC: three"), {"A", "B", "C"})

    def test_parses_concise_and_prefixed_answers(self):
        labels = {"A", "B", "C", "D"}
        self.assertEqual(parse_choice("(B).", labels), "B")
        self.assertEqual(parse_choice("The correct answer is C.", labels), "C")
        self.assertEqual(parse_choice("D because it is last", labels), "D")

    def test_parses_a_letter_followed_by_its_option_text(self):
        # Qwen3-VL-30B-A3B answers MMIU this way, and an earlier delimiter class
        # that closed after ")" discarded every one of these rows.
        labels = {"A", "B", "C", "D"}
        self.assertEqual(parse_choice("B. Yes", labels), "B")
        self.assertEqual(parse_choice("A. Much weaker", labels), "A")
        self.assertEqual(parse_choice("C.Clearer", labels), "C")
        self.assertEqual(parse_choice("D: much better", labels), "D")
        self.assertEqual(parse_choice("B) Sharper", labels), "B")
        self.assertEqual(parse_choice("[C]", labels), "C")

    def test_rejects_invalid_or_ambiguous_answers(self):
        self.assertIsNone(parse_choice("E", {"A", "B", "C", "D"}))
        self.assertIsNone(parse_choice("I cannot tell", {"A", "B", "C", "D"}))
        self.assertIsNone(parse_choice("E. Not an offered option", {"A", "B", "C"}))


DATASET = (
    {"task": "video_captioning", "options": "A: one\nB: two\nC: three"},
    {"task": "video_captioning", "options": "A: one\nB: two"},
    {"task": "temporal_ordering", "options": "A: one\nB: two"},
)


class ReparseTest(unittest.TestCase):
    dataset = DATASET

    def record(self, index, **overrides):
        base = {
            "index": index,
            "task": self.dataset[index]["task"],
            "success": True,
            "prediction": "B. two",
            "choice": None,
        }
        return {**base, **overrides}

    def test_recovers_choices_without_touching_the_stored_records(self):
        records = [self.record(0)]
        reparsed = reparse_records(records, self.dataset)
        self.assertEqual(reparsed[0]["choice"], "B")
        self.assertIsNone(records[0]["choice"])

    def test_keeps_failures_and_answers_outside_the_options(self):
        records = [
            self.record(0, success=False, prediction=None),
            self.record(1, prediction="C. three"),
        ]
        reparsed = reparse_records(records, self.dataset)
        self.assertIsNone(reparsed[0]["choice"])
        self.assertIsNone(reparsed[1]["choice"])

    def test_rejects_results_from_another_dataset(self):
        with self.assertRaises(SystemExit):
            reparse_records([self.record(0, task="temporal_ordering")], self.dataset)
        with self.assertRaises(SystemExit):
            reparse_records([self.record(0) | {"index": 99}], self.dataset)


class PromptTest(unittest.TestCase):
    def test_uses_official_question_first_exception(self):
        row = {"task": "person_reid", "question": "question", "context": "context"}
        self.assertTrue(build_prompt(row).startswith("question\ncontext"))

    def test_defaults_to_context_first(self):
        row = {"task": "other", "question": "question", "context": "context"}
        self.assertTrue(build_prompt(row).startswith("context\nquestion"))


class ScoringTest(unittest.TestCase):
    def test_macro_average_is_unweighted_across_tasks(self):
        records = [
            {
                "index": 0,
                "task": "large",
                "ground_truth": "A",
                "choice": "A",
                "success": True,
            },
            {
                "index": 1,
                "task": "large",
                "ground_truth": "A",
                "choice": "A",
                "success": True,
            },
            {
                "index": 2,
                "task": "small",
                "ground_truth": "B",
                "choice": "A",
                "success": True,
            },
        ]
        score = score_records(records)
        self.assertEqual(score["task_count"], 2)
        self.assertEqual(score["macro_accuracy"], 0.5)

    def test_latest_record_wins_when_retry_succeeds(self):
        records = [
            {
                "index": 0,
                "task": "task",
                "ground_truth": "A",
                "choice": None,
                "success": False,
            },
            {
                "index": 0,
                "task": "task",
                "ground_truth": "A",
                "choice": "A",
                "success": True,
            },
        ]
        score = score_records(records)
        self.assertEqual(score["failures"], 0)
        self.assertEqual(score["macro_accuracy"], 1.0)


class ManifestTest(unittest.TestCase):
    def test_backend_signature_prevents_mixed_resume(self):
        configurations = (
            (
                {"model": "qwen"},
                {"model": "qwen", "backend_signature": "pruned"},
            ),
            (
                {"model": "qwen", "backend_signature": "pruned"},
                {"model": "qwen"},
            ),
        )
        for existing, resumed in configurations:
            with (
                self.subTest(existing=existing),
                TemporaryDirectory() as temporary_directory,
            ):
                output = Path(temporary_directory) / "results.jsonl"
                ensure_manifest(output, existing)

                with self.assertRaises(SystemExit):
                    ensure_manifest(output, resumed)


class ModelCoverageTest(unittest.TestCase):
    def test_rejects_llava_rows_that_cannot_fit_the_32k_context(self):
        dataset = (
            {"input_image_path": ["image.jpg"] * 57},
            {"input_image_path": ["image.jpg"]},
        )
        with self.assertRaisesRegex(SystemExit, "57 images"):
            validate_model_coverage(dataset, [0, 1], "llava-next")

    def test_accepts_a_supported_llava_subset_and_other_families(self):
        dataset = ({"input_image_path": ["image.jpg"] * 8},)
        validate_model_coverage(dataset, [0], "llava-next")
        validate_model_coverage(dataset, [0], "qwen")

    def test_rejects_llava_anyres_rows_above_the_conservative_limit(self):
        dataset = ({"input_image_path": ["image.jpg"] * 9},)
        with self.assertRaisesRegex(SystemExit, "9 images"):
            validate_model_coverage(dataset, [0], "llava-next")


if __name__ == "__main__":
    unittest.main()
