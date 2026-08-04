import tempfile
import unittest
from pathlib import Path

from scripts.patch_image_pruning_encoder import (
    ORIGINAL,
    PATCH_ID,
    REPLACEMENT,
    TARGET,
    apply,
    patch_source,
)

# The guard as it appears in the pinned vLLM revision.
PINNED_SOURCE = """\
class GPUModelRunner:
    def _execute_mm_encoder(self, mm_kwargs):
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
        ):
            if (
                (
                    self.is_multimodal_pruning_enabled
                    or self.requires_sequential_video_encoding
                )
                and modality == "video"
                and num_items > 1
            ):
                pass
"""


class PatchSourceTest(unittest.TestCase):
    def test_extends_sequential_encoding_to_images(self):
        patched = patch_source(PINNED_SOURCE)
        assert patched is not None
        self.assertIn('and modality in ("video", "image")', patched)
        self.assertNotIn('and modality == "video"\n', patched)
        self.assertIn(PATCH_ID, patched)

    def test_reports_already_patched_source(self):
        patched = patch_source(PINNED_SOURCE)
        assert patched is not None
        self.assertIsNone(patch_source(patched))

    def test_rejects_unexpected_source(self):
        with self.assertRaises(SystemExit):
            patch_source(PINNED_SOURCE.replace(ORIGINAL, ""))
        with self.assertRaises(SystemExit):
            patch_source(PINNED_SOURCE + PINNED_SOURCE)


class PatchFileTest(unittest.TestCase):
    def test_applies_verifies_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / TARGET
            target.parent.mkdir(parents=True)
            target.write_text(PINNED_SOURCE, encoding="utf-8")

            with self.assertRaises(SystemExit):
                apply(root, verify_only=True)

            apply(root, verify_only=False)
            first = target.read_text(encoding="utf-8")
            self.assertIn(REPLACEMENT, first)

            apply(root, verify_only=False)
            self.assertEqual(target.read_text(encoding="utf-8"), first)
            apply(root, verify_only=True)

    def test_requires_the_target_file(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit):
            apply(Path(directory), verify_only=False)


if __name__ == "__main__":
    unittest.main()
