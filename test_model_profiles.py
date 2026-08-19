import unittest

from mmiu_eval import LLAVA_NEXT_MAX_IMAGE_TOKENS, LLAVA_NEXT_MIN_IMAGE_TOKENS
from model_profiles import (
    LLAVA_NEXT_GRID_PINPOINTS,
    adapt_messages,
    chat_template_extra_body,
    llava_next_image_tokens,
    resolve_model_family,
    select_best_resolution,
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


class AnyResTokenTest(unittest.TestCase):
    def test_matches_the_transformers_processor(self):
        # Values follow LlavaNextProcessor._get_number_of_features for the
        # pinned checkpoint's grid, so a row's cost is known before inference.
        expected = {
            (336, 336): 1176,
            (224, 224): 1176,
            (512, 512): 2928,
            (1024, 1024): 2928,
            (480, 640): 2340,
            (640, 480): 2352,
            (720, 1280): 1948,
            (300, 900): 2328,
            (900, 300): 2376,
            (256, 1024): 1890,
        }
        for (height, width), tokens in expected.items():
            with self.subTest(size=(height, width)):
                self.assertEqual(llava_next_image_tokens(height, width), tokens)

    def test_selects_the_documented_grid_resolutions(self):
        self.assertEqual(select_best_resolution((336, 336)), (336, 672))
        self.assertEqual(select_best_resolution((512, 512)), (672, 672))
        self.assertEqual(select_best_resolution((256, 1024)), (336, 1008))
        for resolution in LLAVA_NEXT_GRID_PINPOINTS:
            self.assertIn(select_best_resolution(resolution), LLAVA_NEXT_GRID_PINPOINTS)

    def test_bounds_used_to_skip_measurement_hold(self):
        sizes = [
            (height, width)
            for height in (64, 200, 336, 512, 900, 1440, 2160)
            for width in (64, 200, 336, 512, 900, 1440, 2160)
        ]
        for height, width in sizes:
            with self.subTest(size=(height, width)):
                tokens = llava_next_image_tokens(height, width)
                self.assertGreaterEqual(tokens, LLAVA_NEXT_MIN_IMAGE_TOKENS)
                self.assertLessEqual(tokens, LLAVA_NEXT_MAX_IMAGE_TOKENS)

    def test_rejects_degenerate_sizes(self):
        with self.assertRaises(ValueError):
            llava_next_image_tokens(0, 100)


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
