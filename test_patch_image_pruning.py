import tempfile
import unittest
from pathlib import Path

from scripts.patch_image_pruning import (
    MOE_IMAGE_PRUNING_RATE,
    PATCH_ID,
    PATCHES,
    SEQUENTIAL_IMAGE_ENCODING,
    apply,
    patch_source,
)

# The guards as they appear in the pinned vLLM revision.
PINNED_ENCODER = """\
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

PINNED_MOE = """\
class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        super(Qwen3VLForConditionalGeneration, self).__init__()
        multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
"""

SOURCES = {
    SEQUENTIAL_IMAGE_ENCODING.identifier: PINNED_ENCODER,
    MOE_IMAGE_PRUNING_RATE.identifier: PINNED_MOE,
}


def build_root(directory: Path) -> Path:
    for patch in PATCHES:
        target = directory / patch.target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(SOURCES[patch.identifier], encoding="utf-8")
    return directory


class PatchSourceTest(unittest.TestCase):
    def test_extends_sequential_encoding_to_images(self):
        patched = patch_source(PINNED_ENCODER, SEQUENTIAL_IMAGE_ENCODING)
        assert patched is not None
        self.assertIn('and modality in ("video", "image")', patched)
        self.assertNotIn('and modality == "video"\n', patched)

    def test_gives_the_moe_checkpoint_its_image_pruning_rate(self):
        patched = patch_source(PINNED_MOE, MOE_IMAGE_PRUNING_RATE)
        assert patched is not None
        self.assertIn(
            "self.image_pruning_rate = multimodal_config.image_pruning_rate", patched
        )
        # The attribute every inherited image path reads must be assigned before
        # the flag that gates pruning.
        self.assertLess(
            patched.index("self.image_pruning_rate"),
            patched.index("self.is_multimodal_pruning_enabled"),
        )

    def test_reports_already_patched_sources(self):
        for patch in PATCHES:
            with self.subTest(patch=patch.identifier):
                patched = patch_source(SOURCES[patch.identifier], patch)
                assert patched is not None
                self.assertIsNone(patch_source(patched, patch))

    def test_rejects_unexpected_sources(self):
        for patch in PATCHES:
            source = SOURCES[patch.identifier]
            with self.subTest(patch=patch.identifier):
                with self.assertRaises(SystemExit):
                    patch_source(source.replace(patch.original, ""), patch)
                with self.assertRaises(SystemExit):
                    patch_source(source + source, patch)


class PatchFileTest(unittest.TestCase):
    def test_applies_every_patch_verifies_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_root(Path(directory))

            with self.assertRaises(SystemExit):
                apply(root, verify_only=True)

            apply(root, verify_only=False)
            first = {
                patch.identifier: (root / patch.target).read_text(encoding="utf-8")
                for patch in PATCHES
            }
            for patch in PATCHES:
                self.assertIn(patch.replacement, first[patch.identifier])

            apply(root, verify_only=False)
            for patch in PATCHES:
                self.assertEqual(
                    (root / patch.target).read_text(encoding="utf-8"),
                    first[patch.identifier],
                )
            apply(root, verify_only=True)

    def test_requires_every_target_file(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit):
            apply(Path(directory), verify_only=False)

    def test_identifier_names_every_patch(self):
        for patch in PATCHES:
            self.assertIn(patch.identifier, PATCH_ID)


if __name__ == "__main__":
    unittest.main()
