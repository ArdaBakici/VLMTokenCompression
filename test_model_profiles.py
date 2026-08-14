import unittest

from model_profiles import (
    adapt_messages,
    chat_template_extra_body,
    resolve_model_family,
)


class ModelFamilyTest(unittest.TestCase):
    def test_recognizes_supported_models(self):
        expected = {
            "Qwen/Qwen3-VL-8B-Instruct": "qwen",
            "OpenGVLab/InternVL3-8B-hf": "internvl",
            "llava-hf/llava-v1.6-mistral-7b-hf": "llava-next",
            "local/model": "generic",
        }
        for model, family in expected.items():
            with self.subTest(model=model):
                self.assertEqual(resolve_model_family(model), family)

    def test_only_qwen_receives_thinking_template_arguments(self):
        self.assertEqual(
            chat_template_extra_body("qwen", False),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertIsNone(chat_template_extra_body("internvl", False))
        self.assertIsNone(chat_template_extra_body("llava-next", False))
        with self.assertRaises(ValueError):
            chat_template_extra_body("internvl", True)


class MessageAdaptationTest(unittest.TestCase):
    def test_folds_llava_system_text_without_reordering_media(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/jpeg,x"}}
        messages = [
            {"role": "system", "content": "Analyze the video."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Video 1:"},
                    image,
                    {"type": "text", "text": "Question"},
                ],
            },
        ]

        adapted = adapt_messages(messages, "llava-next")

        self.assertEqual([message["role"] for message in adapted], ["user"])
        self.assertEqual(adapted[0]["content"][0]["text"], "Analyze the video.\n\n")
        self.assertEqual(adapted[0]["content"][2], image)
        self.assertEqual(messages[0]["role"], "system")

    def test_preserves_official_messages_for_other_models(self):
        messages = [{"role": "system", "content": "System"}]
        self.assertIs(adapt_messages(messages, "qwen"), messages)
        self.assertIs(adapt_messages(messages, "internvl"), messages)


if __name__ == "__main__":
    unittest.main()
