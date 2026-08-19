import tempfile
import unittest
from pathlib import Path

from scripts.patch_qwen_multi_image import (
    HIPRUNE_QWEN_MULTI_IMAGE_CACHE_POSITION,
    HIPRUNE_QWEN_MULTI_IMAGE_CACHE_RESYNC,
    HIPRUNE_QWEN_MULTI_IMAGE_MASK,
    PATCH_SETS,
    VISIONZIP_QWEN_MULTI_IMAGE_CACHE_POSITION,
    VISIONZIP_QWEN_MULTI_IMAGE_CACHE_RESYNC,
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
        
            # print(f"Visual tokens: {visual_token_num}")
        ###################
        
        outputs = self.model(
            cache_position=cache_position,
        )

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
    ):
        return model_inputs
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
            del contextual_tokens, hidden_states_filtered, hidden_to_merge,aggregated_hidden

        outputs = self.model(
            cache_position=cache_position,
        )

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
    ):
        return model_inputs
"""

SOURCES = {
    HIPRUNE_QWEN_MULTI_IMAGE_MASK.identifier: PINNED_HIPRUNE,
    HIPRUNE_QWEN_MULTI_IMAGE_CACHE_POSITION.identifier: PINNED_HIPRUNE,
    HIPRUNE_QWEN_MULTI_IMAGE_CACHE_RESYNC.identifier: PINNED_HIPRUNE,
    VISIONZIP_QWEN_MULTI_IMAGE_MASK.identifier: PINNED_VISIONZIP,
    VISIONZIP_QWEN_MULTI_IMAGE_GATHER.identifier: PINNED_VISIONZIP,
    VISIONZIP_QWEN_MULTI_IMAGE_CACHE_POSITION.identifier: PINNED_VISIONZIP,
    VISIONZIP_QWEN_MULTI_IMAGE_CACHE_RESYNC.identifier: PINNED_VISIONZIP,
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

    def test_rebuilds_hiprune_cache_position_after_pruning(self):
        patched = patch_source(PINNED_HIPRUNE, HIPRUNE_QWEN_MULTI_IMAGE_CACHE_POSITION)
        assert patched is not None
        self.assertIn(
            "cache_position = torch.arange(\n"
            "                inputs_embeds.shape[1], device=inputs_embeds.device\n"
            "            )",
            patched,
        )
        # Must come after inputs_embeds is pruned, not before.
        self.assertLess(
            patched.index("inputs_embeds = inputs_embeds[:, img_mask]"),
            patched.index("cache_position = torch.arange("),
        )

    def test_rebuilds_visionzip_cache_position_after_pruning(self):
        patched = patch_source(
            PINNED_VISIONZIP, VISIONZIP_QWEN_MULTI_IMAGE_CACHE_POSITION
        )
        assert patched is not None
        self.assertIn(
            "cache_position = torch.arange(\n"
            "                inputs_embeds.shape[1], device=inputs_embeds.device\n"
            "            )",
            patched,
        )
        self.assertLess(
            patched.index("inputs_embeds = inputs_embeds[:, img_mask]"),
            patched.index("cache_position = torch.arange("),
        )

    def test_hiprune_cache_resync_overrides_the_generation_hook(self):
        patched = patch_source(PINNED_HIPRUNE, HIPRUNE_QWEN_MULTI_IMAGE_CACHE_RESYNC)
        assert patched is not None
        self.assertIn("def _update_model_kwargs_for_generation(", patched)
        self.assertIn("cached_length = past_key_values.get_seq_length()", patched)
        # The override must sit before prepare_inputs_for_generation and call
        # super() rather than reimplementing the base bookkeeping.
        self.assertLess(
            patched.index("def _update_model_kwargs_for_generation("),
            patched.index("def prepare_inputs_for_generation("),
        )
        self.assertIn("super()._update_model_kwargs_for_generation(", patched)

    def test_visionzip_cache_resync_overrides_the_generation_hook(self):
        patched = patch_source(
            PINNED_VISIONZIP, VISIONZIP_QWEN_MULTI_IMAGE_CACHE_RESYNC
        )
        assert patched is not None
        self.assertIn("def _update_model_kwargs_for_generation(", patched)
        self.assertLess(
            patched.index("def _update_model_kwargs_for_generation("),
            patched.index("def prepare_inputs_for_generation("),
        )
        self.assertIn("super()._update_model_kwargs_for_generation(", patched)

    def test_cache_resync_recovers_the_true_cache_length(self):
        # Exercises the actual arithmetic the patched method runs, independent
        # of torch: a stand-in past_key_values/attention_mask/cache_position
        # walked through a prefill (pruned) step and two decode steps.
        class Cache:
            def __init__(self, length):
                self.length = length

            def get_seq_length(self):
                return self.length

        class Mask(list):
            @property
            def shape(self):
                return (1, len(self))

            def new_ones(self, shape):
                return Mask([1] * shape[1])

        class CachePosition(list):
            @property
            def shape(self):
                return (len(self),)

            device = "cpu"

        def arange(a, b):
            return CachePosition(range(a, b))

        def default_super_update(model_kwargs, num_new_tokens):
            model_kwargs["attention_mask"] = Mask(
                model_kwargs["attention_mask"] + [1] * num_new_tokens
            )
            last = model_kwargs["cache_position"][-1]
            model_kwargs["cache_position"] = CachePosition([last + num_new_tokens])
            return model_kwargs

        def patched_update(true_cache, model_kwargs):
            model_kwargs = default_super_update(model_kwargs, num_new_tokens=1)
            model_kwargs["past_key_values"] = true_cache
            cached_length = true_cache.get_seq_length()
            upcoming = model_kwargs["cache_position"].shape[0]
            if model_kwargs["attention_mask"].shape[-1] != cached_length + upcoming:
                model_kwargs["attention_mask"] = model_kwargs["attention_mask"].new_ones(
                    (1, cached_length + upcoming)
                )
                model_kwargs["cache_position"] = arange(
                    cached_length, cached_length + upcoming
                )
            return model_kwargs

        original_length, pruned_length = 1420, 413
        model_kwargs = {
            "attention_mask": Mask([1] * original_length),
            "cache_position": arange(0, original_length),
        }

        model_kwargs = patched_update(Cache(pruned_length), model_kwargs)
        self.assertEqual(model_kwargs["attention_mask"].shape[-1], pruned_length + 1)
        self.assertEqual(list(model_kwargs["cache_position"]), [pruned_length])

        model_kwargs = patched_update(Cache(pruned_length + 1), model_kwargs)
        self.assertEqual(model_kwargs["attention_mask"].shape[-1], pruned_length + 2)
        self.assertEqual(list(model_kwargs["cache_position"]), [pruned_length + 1])

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
