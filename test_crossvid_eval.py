import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from crossvid_eval import (
    extract_json_object,
    parse_choices,
    parse_interval_prediction,
    parse_sequence,
    run_task,
)


class OutputParsingTest(unittest.TestCase):
    def test_single_and_multiple_choices(self):
        labels = ["A", "B", "C", "D"]
        self.assertEqual(parse_choices("The answer is (C).", labels, False), "C")
        self.assertEqual(parse_choices("<answer>D, B</answer>", labels, True), "BD")

    def test_rejects_verbose_or_unknown_choices(self):
        labels = ["A", "B", "C", "D"]
        self.assertIsNone(parse_choices("C because video B differs", labels, False))
        self.assertIsNone(parse_choices("AE", labels, True))

    def test_sequence_is_canonicalized_and_validated(self):
        self.assertEqual(parse_sequence("2, 3, 1", 3), "2->3->1")
        self.assertIsNone(parse_sequence("1->1->2", 3))

    def test_interval_is_parsed_and_ordered(self):
        self.assertEqual(parse_interval_prediction("(1.5, 3)"), [1.5, 3.0])
        self.assertIsNone(parse_interval_prediction("3, 1"))

    def test_extracts_json_from_judge_wrapper(self):
        value = extract_json_object(
            '<score>{"coverage": [true], "correctness": [false]}</score>'
        )
        self.assertEqual(value["coverage"], [True])


class RunnerTest(unittest.TestCase):
    @patch("crossvid_eval.make_client", return_value=object())
    @patch("crossvid_eval.load_official_module")
    def test_exports_resumable_official_result(self, load_module, _make_client):
        module = SimpleNamespace()
        module.evaluate = Mock(
            side_effect=lambda pair, max_tries: (pair, "The answer is (B).")
        )
        load_module.return_value = module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qa_dir = root / "QA"
            qa_dir.mkdir()
            pair = {
                "id": 7,
                "question": "Question?",
                "options": ["A. First", "B. Second"],
                "videos": ["one.mp4", "two.mp4"],
                "answer": "B",
            }
            (qa_dir / "CC.json").write_text(json.dumps([pair]))
            args = argparse.Namespace(
                qa_dir=qa_dir,
                start=0,
                limit=None,
                vendor_root=root / "vendor",
                model="test-model",
                frames=8,
                length=64,
                uav_root=root / "uav",
                video_root=root / "videos",
                results_dir=root / "results",
                base_url="http://localhost/v1",
                max_tokens=16,
                enable_thinking=False,
                workers=1,
                timeout=10,
                retries=0,
            )

            run_task("CC", args)
            run_task("CC", args)
            exported = json.loads(
                (args.results_dir / "CC_result.json").read_text()
            )
            self.assertEqual(exported[0]["answer"], "B")
            self.assertTrue(exported[0]["success"])
            self.assertNotIn("efficiency", exported[0])
            module.evaluate.assert_called_once()

    @patch("crossvid_eval.stream_chat_completion")
    @patch("crossvid_eval.make_client", return_value=object())
    @patch("crossvid_eval.load_official_module")
    def test_adapts_llava_request_and_preserves_media_order(
        self, load_module, _make_client, stream_completion
    ):
        module = SimpleNamespace()
        image = {"type": "image_url", "image_url": {"url": "data:image/jpeg,x"}}
        messages = [
            {"role": "system", "content": "You are a helpful video analyzer."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Video 1:"},
                    image,
                    {"type": "text", "text": "Question?"},
                ],
            },
        ]

        def evaluate(pair, max_tries):
            return pair, module.chat(messages)

        module.evaluate = evaluate
        load_module.return_value = module
        stream_completion.return_value = (
            "B",
            {"schema_version": 1, "prompt_tokens": 10},
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qa_dir = root / "QA"
            qa_dir.mkdir()
            pair = {
                "id": 7,
                "question": "Question?",
                "options": ["A. First", "B. Second"],
                "videos": ["one.mp4", "two.mp4"],
                "answer": "B",
            }
            (qa_dir / "CC.json").write_text(json.dumps([pair]))
            args = argparse.Namespace(
                qa_dir=qa_dir,
                start=0,
                limit=None,
                vendor_root=root / "vendor",
                model="llava-hf/llava-v1.6-mistral-7b-hf",
                model_family="auto",
                frames=8,
                length=64,
                uav_root=root / "uav",
                video_root=root / "videos",
                results_dir=root / "results",
                base_url="http://localhost/v1",
                max_tokens=16,
                enable_thinking=False,
                workers=1,
                timeout=10,
                retries=0,
                backend_signature="test",
            )

            run_task("CC", args)

        request = stream_completion.call_args.args[1]
        self.assertNotIn("extra_body", request)
        self.assertEqual([message["role"] for message in request["messages"]], ["user"])
        content = request["messages"][0]["content"]
        self.assertEqual(content[0]["text"], "You are a helpful video analyzer.\n\n")
        self.assertEqual(content[2], image)


if __name__ == "__main__":
    unittest.main()
