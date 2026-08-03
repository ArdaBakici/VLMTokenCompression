import unittest

from mmiu_eval import build_prompt, option_labels, parse_choice, score_records


class ChoiceParsingTest(unittest.TestCase):
    def test_finds_option_labels(self):
        self.assertEqual(option_labels("A: one\nB: two\nC: three"), {"A", "B", "C"})

    def test_parses_concise_and_prefixed_answers(self):
        labels = {"A", "B", "C", "D"}
        self.assertEqual(parse_choice("(B).", labels), "B")
        self.assertEqual(parse_choice("The correct answer is C.", labels), "C")
        self.assertEqual(parse_choice("D because it is last", labels), "D")

    def test_rejects_invalid_or_ambiguous_answers(self):
        self.assertIsNone(parse_choice("E", {"A", "B", "C", "D"}))
        self.assertIsNone(parse_choice("I cannot tell", {"A", "B", "C", "D"}))


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


if __name__ == "__main__":
    unittest.main()
