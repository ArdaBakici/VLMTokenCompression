import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
LAUNCHER = ROOT / "scripts" / "run_benchmark.sh"


def launcher_profile(benchmark: str, model: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("MODEL", None)
    environment.pop("MODEL_FAMILY", None)
    environment.pop("MAX_MODEL_LEN", None)
    environment.pop("FRAMES", None)
    environment["PRINT_MODEL_PROFILE"] = "1"
    output = subprocess.run(
        ["bash", str(LAUNCHER), benchmark, model],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return dict(line.split("=", 1) for line in output.splitlines())


class LauncherProfileTest(unittest.TestCase):
    def test_native_internvl_profile(self):
        profile = launcher_profile("crossvid", "OpenGVLab/InternVL3-8B-hf")
        self.assertEqual(profile["model_family"], "internvl")
        self.assertEqual(profile["max_model_len"], "32768")
        self.assertEqual(profile["crossvid_frames"], "16")
        self.assertEqual(profile["mm_processor_kwargs"], '{"max_patches":1}')

    def test_llava_profile_preserves_multimodal_interleaving(self):
        profile = launcher_profile(
            "crossvid", "llava-hf/llava-v1.6-mistral-7b-hf"
        )
        self.assertEqual(profile["model_family"], "llava-next")
        self.assertEqual(profile["max_model_len"], "32768")
        self.assertEqual(profile["crossvid_frames"], "8")
        self.assertEqual(profile["interleave_mm_strings"], "1")
        self.assertTrue(profile["chat_template"].endswith("llava_next_interleaved.jinja"))

    def test_llava_template_renders_media_in_content_order(self):
        template = (
            ROOT / "chat_templates" / "llava_next_interleaved.jinja"
        ).read_text()
        self.assertIn("{% for item in message['content'] %}", template)
        self.assertNotIn("selectattr", template)
        image_branch = template.index("item['type'] == 'image'")
        text_branch = template.index("item['type'] == 'text'")
        loop_end = template.index("{% endfor %}", image_branch)
        self.assertLess(image_branch, loop_end)
        self.assertLess(text_branch, loop_end)

    def test_rejects_the_original_internvl_format(self):
        environment = os.environ.copy()
        environment.pop("MODEL", None)
        environment["PRINT_MODEL_PROFILE"] = "1"
        result = subprocess.run(
            ["bash", str(LAUNCHER), "mmiu", "OpenGVLab/InternVL3-8B"],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("InternVL3-8B-hf", result.stderr)


if __name__ == "__main__":
    unittest.main()
