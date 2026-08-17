import tempfile
import unittest
from pathlib import Path

from scripts.patch_qwen_multi_image import (
    HIPRUNE_QWEN_MULTI_IMAGE_MASK,
    PATCH_SETS,
    VISIONZIP_QWEN_MULTI_IMAGE_GATHER,
    VISIONZIP_QWEN_MULTI_IMAGE_MASK,
    apply,
    patch_id,
    patch_source,
)

# The anchors as they appear in the pinned revisions (verbatim, including the
# trailing whitespace the released files themselves contain).
PINNED_HIPRUNE = """\
class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    def forward(self, input_ids=None, pixel_values=None, image_grid_thw=None):
        ##### HiPrune #####
        if inputs_embeds.shape[1] != 1: # KV cache
            select_mask = torch.zeros_like(deep_attention, dtype=torch.bool)
            select_mask[retain] = True

            img_mask = (input_ids == self.config.image_token_id)[0]  
            st_idx = torch.nonzero(img_mask, as_tuple=True)[0]         

            if st_idx.numel() > 0:
                first, last = st_idx[0].item(), st_idx[-1].item()     
                img_mask[first:last+1] = ~select_mask
                img_mask = ~img_mask

            position_ids = position_ids[:,:,img_mask]
            attention_mask = attention_mask[:, img_mask]
            inputs_embeds = inputs_embeds[:, img_mask]
        ###################
"""

PINNED_VISIONZIP = """\
class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    def forward(self, input_ids=None, pixel_values=None, image_grid_thw=None):
        if select_pixel:
            select_mask = torch.zeros_like(attn_logits, dtype=torch.bool)
            select_mask[topk_indices] = True

            false_pos = (~select_mask).nonzero(as_tuple=True)[0]   

            select_mask[false_pos[target_indices]] = True


            img_mask = (input_ids == self.config.image_token_id)[0]  
            st_idx = torch.nonzero(img_mask, as_tuple=True)[0]         

            if st_idx.numel() > 0:
                first, last = st_idx[0].item(), st_idx[-1].item()     
                img_mask[first:last+1] = ~select_mask
                img_mask = ~img_mask
                contexual_input_idx = false_pos[target_indices]+first

            hidden_states_filtered = inputs_embeds[:, first:last+1][:,contextual_mask]
            hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1]), target_indices), :]

            position_ids = position_ids[:,:,img_mask]
            attention_mask = attention_mask[:, img_mask]
            inputs_embeds[:,contexual_input_idx] =  contextual_tokens
            inputs_embeds = inputs_embeds[:, img_mask]
"""

SOURCES = {
    HIPRUNE_QWEN_MULTI_IMAGE_MASK.identifier: PINNED_HIPRUNE,
    VISIONZIP_QWEN_MULTI_IMAGE_MASK.identifier: PINNED_VISIONZIP,
    VISIONZIP_QWEN_MULTI_IMAGE_GATHER.identifier: PINNED_VISIONZIP,
}


def build_root(directory: Path, method: str) -> Path:
    sources = {"hiprune-qwen": PINNED_HIPRUNE, "visionzip-qwen": PINNED_VISIONZIP}
    for patch in PATCH_SETS[method]:
        target = directory / patch.target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sources[method], encoding="utf-8")
    return directory


class PatchSourceTest(unittest.TestCase):
    def test_indexes_hiprune_mask_assignment_through_st_idx(self):
        patched = patch_source(PINNED_HIPRUNE, HIPRUNE_QWEN_MULTI_IMAGE_MASK)
        assert patched is not None
        self.assertIn("img_mask[st_idx] = ~select_mask", patched)
        self.assertNotIn("img_mask[first:last+1] = ~select_mask", patched)
        # The rest of the block (still using first/last, and the subsequent
        # boolean-mask filtering of position_ids/attention_mask/inputs_embeds)
        # must be untouched.
        self.assertIn("first, last = st_idx[0].item(), st_idx[-1].item()", patched)
        self.assertIn("position_ids = position_ids[:,:,img_mask]", patched)

    def test_indexes_visionzip_mask_assignment_and_scatter_target(self):
        patched = patch_source(PINNED_VISIONZIP, VISIONZIP_QWEN_MULTI_IMAGE_MASK)
        assert patched is not None
        self.assertIn("img_mask[st_idx] = ~select_mask", patched)
        self.assertIn(
            "contexual_input_idx = st_idx[false_pos[target_indices]]", patched
        )
        self.assertNotIn("img_mask[first:last+1] = ~select_mask", patched)
        self.assertNotIn("false_pos[target_indices]+first", patched)

    def test_gathers_visionzip_hidden_states_through_st_idx(self):
        patched = patch_source(PINNED_VISIONZIP, VISIONZIP_QWEN_MULTI_IMAGE_GATHER)
        assert patched is not None
        self.assertIn(
            "hidden_states_filtered = inputs_embeds[:, st_idx][:,contextual_mask]",
            patched,
        )
        self.assertNotIn(
            "hidden_states_filtered = inputs_embeds[:, first:last+1][:,contextual_mask]",
            patched,
        )

    def test_reports_already_patched_sources(self):
        for identifier, source in SOURCES.items():
            patch = next(
                p for patches in PATCH_SETS.values() for p in patches
                if p.identifier == identifier
            )
            with self.subTest(patch=identifier):
                patched = patch_source(source, patch)
                assert patched is not None
                self.assertIsNone(patch_source(patched, patch))

    def test_rejects_unexpected_sources(self):
        for identifier, source in SOURCES.items():
            patch = next(
                p for patches in PATCH_SETS.values() for p in patches
                if p.identifier == identifier
            )
            with self.subTest(patch=identifier):
                with self.assertRaises(SystemExit):
                    patch_source(source.replace(patch.original, ""), patch)
                with self.assertRaises(SystemExit):
                    patch_source(source + source, patch)


class PatchFileTest(unittest.TestCase):
    def test_applies_hiprune_verifies_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_root(Path(directory), "hiprune-qwen")

            with self.assertRaises(SystemExit):
                apply(root, "hiprune-qwen", verify_only=True)

            apply(root, "hiprune-qwen", verify_only=False)
            path = root / HIPRUNE_QWEN_MULTI_IMAGE_MASK.target
            first = path.read_text(encoding="utf-8")
            self.assertIn(HIPRUNE_QWEN_MULTI_IMAGE_MASK.replacement, first)

            apply(root, "hiprune-qwen", verify_only=False)
            self.assertEqual(path.read_text(encoding="utf-8"), first)
            apply(root, "hiprune-qwen", verify_only=True)

    def test_applies_visionzip_verifies_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_root(Path(directory), "visionzip-qwen")

            with self.assertRaises(SystemExit):
                apply(root, "visionzip-qwen", verify_only=True)

            apply(root, "visionzip-qwen", verify_only=False)
            path = root / VISIONZIP_QWEN_MULTI_IMAGE_MASK.target
            first = path.read_text(encoding="utf-8")
            self.assertIn(VISIONZIP_QWEN_MULTI_IMAGE_MASK.replacement, first)
            self.assertIn(VISIONZIP_QWEN_MULTI_IMAGE_GATHER.replacement, first)

            apply(root, "visionzip-qwen", verify_only=False)
            self.assertEqual(path.read_text(encoding="utf-8"), first)
            apply(root, "visionzip-qwen", verify_only=True)

    def test_requires_every_target_file(self):
        for method in PATCH_SETS:
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(
                SystemExit
            ):
                apply(Path(directory), method, verify_only=False)

    def test_patch_id_names_every_patch_for_its_method(self):
        for method, patches in PATCH_SETS.items():
            identifier = patch_id(method)
            for patch in patches:
                self.assertIn(patch.identifier, identifier)


if __name__ == "__main__":
    unittest.main()
