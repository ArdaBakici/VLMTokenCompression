"""Apply the local multi-image indexing fix that hiprune-qwen/visionzip-qwen need.

HiPrune's and VisionZip's released Qwen2.5-VL forks (Qwen2_5_VL/qwen2_5_vl_HiPrune.py,
Qwen2_5_VL/qwen2_5vl_visionzip.py) each locate the image tokens to prune with:

    img_mask = (input_ids == self.config.image_token_id)[0]
    st_idx = torch.nonzero(img_mask, as_tuple=True)[0]
    first, last = st_idx[0].item(), st_idx[-1].item()
    img_mask[first:last+1] = ~select_mask

`select_mask` is sized to the true number of image tokens (summed across every
image in the request). `img_mask[first:last+1]` is only the same size when
that is a *single* image: `<|vision_start|>`/`<|vision_end|>` separators
between multiple images sit inside `[first, last]` too, so the slice is
longer than `select_mask` and the assignment shape-mismatches --
`RuntimeError: The expanded size of the tensor (N) must match the existing
size (M) at non-singleton dimension 0`. Both forks were released and
evaluated single-image only (VisionZip's own results table only reports
single-image benchmarks); this is not a caller-side configuration issue.

`st_idx` already holds the exact, possibly-gapped image-token positions in
the same left-to-right order `select_mask` (and, in VisionZip's case,
`false_pos`/`contextual_mask`) index -- the vision tower necessarily produces
its output in that same order for `masked_scatter` to work at all, a few
lines earlier in the same unpatched function. Indexing through `st_idx`
instead of `first:last+1` is therefore a no-op for one image (where the range
is already contiguous) and the direct fix for more than one: neither
method's token-selection logic (which tokens to keep, and why) is touched,
only how the selection is written back into a sequence that is not one
uninterrupted image block.

Fixing that reveals a second, unrelated bug that both forks also share and
that has nothing to do with multiple images: `cache_position` is built once,
before either fork's `forward()` runs, from the request's original (unpruned)
token count, and is never touched by the pruning code -- so the very next
line, `self.model(..., cache_position=cache_position, ...)`, builds a causal
mask from the (now shorter) `inputs_embeds` against a `cache_position` still
sized for the original prompt, and shape-mismatches the same way one line
later. Rebuilding `cache_position` as a fresh `torch.arange` matching the
pruned length fixes that forward() call -- but neither fork overrides
`_update_model_kwargs_for_generation`, so HF's stock `GenerationMixin` then
keeps extending `attention_mask`/`cache_position` from the *pre-pruning*
prompt length for every later decode step, while `past_key_values` only ever
held the pruned length: the same crash recurs one generated token later.
This one is not single-image-safe either, and single-image runs of these two
methods would hit it too the moment more than one output token is generated
(MMIU's terse "answer with one letter" prompting likely explains why neither
fork's own reported benchmarks surfaced it). The fix rederives both
`attention_mask` and `cache_position` from `past_key_values`'s actual length
after every step -- a no-op once they already agree, i.e. every step after
the first.

Each patch is applied to the cloned pinned checkout, is idempotent, and
refuses to run when the pinned source no longer matches what it expects.
"""

from __future__ import annotations

import argparse
import os
import py_compile
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Patch:
    identifier: str
    target: Path
    original: str
    replacement: str


HIPRUNE_QWEN_MULTI_IMAGE_MASK = Patch(
    identifier="hiprune-qwen-multi-image-mask-v1",
    target=Path("Qwen2_5_VL") / "qwen2_5_vl_HiPrune.py",
    original=(
        "            if st_idx.numel() > 0:\n"
        "                first, last = st_idx[0].item(), st_idx[-1].item()     \n"
        "                img_mask[first:last+1] = ~select_mask\n"
        "                img_mask = ~img_mask\n"
    ),
    replacement=(
        "            if st_idx.numel() > 0:\n"
        "                first, last = st_idx[0].item(), st_idx[-1].item()     \n"
        "                # Local patch: hiprune-qwen-multi-image-mask-v1. first:last+1\n"
        "                # is only every image token for a single image; st_idx is the\n"
        "                # exact (possibly gapped) image-token positions in the order\n"
        "                # select_mask indexes, correct for one image or many. See\n"
        "                # scripts/patch_qwen_multi_image.py for the full explanation.\n"
        "                img_mask[st_idx] = ~select_mask\n"
        "                img_mask = ~img_mask\n"
    ),
)

VISIONZIP_QWEN_MULTI_IMAGE_MASK = Patch(
    identifier="visionzip-qwen-multi-image-mask-v1",
    target=Path("Qwen2_5_VL") / "qwen2_5vl_visionzip.py",
    original=(
        "            if st_idx.numel() > 0:\n"
        "                first, last = st_idx[0].item(), st_idx[-1].item()     \n"
        "                img_mask[first:last+1] = ~select_mask\n"
        "                img_mask = ~img_mask\n"
        "                contexual_input_idx = false_pos[target_indices]+first\n"
    ),
    replacement=(
        "            if st_idx.numel() > 0:\n"
        "                first, last = st_idx[0].item(), st_idx[-1].item()     \n"
        "                # Local patch: visionzip-qwen-multi-image-mask-v1. Same bug as\n"
        "                # hiprune-qwen-multi-image-mask-v1: first:last+1 is only every\n"
        "                # image token for a single image. st_idx is the exact (possibly\n"
        "                # gapped) image-token positions in the order select_mask and\n"
        "                # false_pos index, needed for both the mask assignment and the\n"
        "                # merged-token scatter target. See\n"
        "                # scripts/patch_qwen_multi_image.py for the full explanation.\n"
        "                img_mask[st_idx] = ~select_mask\n"
        "                img_mask = ~img_mask\n"
        "                contexual_input_idx = st_idx[false_pos[target_indices]]\n"
    ),
)

VISIONZIP_QWEN_MULTI_IMAGE_GATHER = Patch(
    identifier="visionzip-qwen-multi-image-gather-v1",
    target=Path("Qwen2_5_VL") / "qwen2_5vl_visionzip.py",
    original=(
        "            hidden_states_filtered = inputs_embeds[:, first:last+1][:,contextual_mask]\n"
    ),
    replacement=(
        "            # Local patch: visionzip-qwen-multi-image-gather-v1. first:last+1\n"
        "            # is only every image token for a single image (see\n"
        "            # visionzip-qwen-multi-image-mask-v1 above); st_idx gathers exactly\n"
        "            # the image-token positions in the order contextual_mask indexes,\n"
        "            # correct for one image or many.\n"
        "            hidden_states_filtered = inputs_embeds[:, st_idx][:,contextual_mask]\n"
    ),
)

# `cache_position` is built once, before either fork's forward() runs, from
# the request's original (unpruned) token count. Once the mask patches above
# prune position_ids/attention_mask/inputs_embeds down to the retained
# budget, cache_position is the one thing left unpruned: the self.model(...)
# call a few lines later passes it straight through, builds a causal mask
# sized from the (now shorter) inputs_embeds against this stale, longer
# cache_position, and shape-mismatches -- `RuntimeError: The size of tensor a
# (pruned length) must match the size of tensor b (original length) at
# non-singleton dimension 0`. The pruned tokens are the entire sequence the
# KV cache will ever have for this prefill, so, like any other model's first
# forward call, their position for causal-masking and cache-indexing
# purposes is simply their (now dense) index in it, 0..N-1.
HIPRUNE_QWEN_MULTI_IMAGE_CACHE_POSITION = Patch(
    identifier="hiprune-qwen-multi-image-cache-position-v1",
    target=Path("Qwen2_5_VL") / "qwen2_5_vl_HiPrune.py",
    original=(
        "            position_ids = position_ids[:,:,img_mask]\n"
        "            attention_mask = attention_mask[:, img_mask]\n"
        "            inputs_embeds = inputs_embeds[:, img_mask]\n"
        "        \n"
        '            # print(f"Visual tokens: {visual_token_num}")\n'
    ),
    replacement=(
        "            position_ids = position_ids[:,:,img_mask]\n"
        "            attention_mask = attention_mask[:, img_mask]\n"
        "            inputs_embeds = inputs_embeds[:, img_mask]\n"
        "            # Local patch: hiprune-qwen-multi-image-cache-position-v1.\n"
        "            # See scripts/patch_qwen_multi_image.py for the full explanation.\n"
        "            cache_position = torch.arange(\n"
        "                inputs_embeds.shape[1], device=inputs_embeds.device\n"
        "            )\n"
        "        \n"
        '            # print(f"Visual tokens: {visual_token_num}")\n'
    ),
)

VISIONZIP_QWEN_MULTI_IMAGE_CACHE_POSITION = Patch(
    identifier="visionzip-qwen-multi-image-cache-position-v1",
    target=Path("Qwen2_5_VL") / "qwen2_5vl_visionzip.py",
    original=(
        "            position_ids = position_ids[:,:,img_mask]\n"
        "            attention_mask = attention_mask[:, img_mask]\n"
        "            inputs_embeds[:,contexual_input_idx] =  contextual_tokens\n"
        "            inputs_embeds = inputs_embeds[:, img_mask]\n"
        "            del contextual_tokens, hidden_states_filtered, hidden_to_merge,aggregated_hidden\n"
    ),
    replacement=(
        "            position_ids = position_ids[:,:,img_mask]\n"
        "            attention_mask = attention_mask[:, img_mask]\n"
        "            inputs_embeds[:,contexual_input_idx] =  contextual_tokens\n"
        "            inputs_embeds = inputs_embeds[:, img_mask]\n"
        "            # Local patch: visionzip-qwen-multi-image-cache-position-v1.\n"
        "            # Same bug as hiprune-qwen-multi-image-cache-position-v1; see\n"
        "            # scripts/patch_qwen_multi_image.py for the full explanation.\n"
        "            cache_position = torch.arange(\n"
        "                inputs_embeds.shape[1], device=inputs_embeds.device\n"
        "            )\n"
        "            del contextual_tokens, hidden_states_filtered, hidden_to_merge,aggregated_hidden\n"
    ),
)

# The cache-position patches above only fix the forward() call they run in
# (the prefill step where pruning happens). Neither fork overrides
# _update_model_kwargs_for_generation, so HF's stock GenerationMixin keeps
# extending attention_mask/cache_position from the pre-pruning prompt length
# for every later decode step -- it has no way to know forward() quietly
# processed a shorter sequence. That external bookkeeping and the actual
# past_key_values (which only ever held the pruned length) then diverge
# starting with the very first decode step, crashing the same way one token
# later. Since every request this server sends is a single, unpadded sample,
# attention_mask is always all-ones; rederiving both it and cache_position
# from past_key_values.get_seq_length() -- the cache's real length -- after
# every step keeps them correct regardless of how much pruning happened, and
# is a no-op once they already agree (i.e. every step after the first).
UPDATE_MODEL_KWARGS_METHOD = (
    "    def _update_model_kwargs_for_generation(\n"
    "        self,\n"
    "        outputs,\n"
    "        model_kwargs,\n"
    "        is_encoder_decoder: bool = False,\n"
    "        num_new_tokens: int = 1,\n"
    "    ):\n"
    "        # Local patch: {identifier}.\n"
    "        # See scripts/patch_qwen_multi_image.py for the full explanation.\n"
    "        model_kwargs = super()._update_model_kwargs_for_generation(\n"
    "            outputs,\n"
    "            model_kwargs,\n"
    "            is_encoder_decoder=is_encoder_decoder,\n"
    "            num_new_tokens=num_new_tokens,\n"
    "        )\n"
    "        past_key_values = model_kwargs.get(\"past_key_values\")\n"
    "        attention_mask = model_kwargs.get(\"attention_mask\")\n"
    "        cache_position = model_kwargs.get(\"cache_position\")\n"
    "        if (\n"
    "            is_encoder_decoder\n"
    "            or past_key_values is None\n"
    "            or attention_mask is None\n"
    "            or cache_position is None\n"
    "        ):\n"
    "            return model_kwargs\n"
    "        cached_length = past_key_values.get_seq_length()\n"
    "        upcoming = cache_position.shape[0]\n"
    "        if attention_mask.shape[-1] == cached_length + upcoming:\n"
    "            return model_kwargs\n"
    "        model_kwargs[\"attention_mask\"] = attention_mask.new_ones(\n"
    "            (attention_mask.shape[0], cached_length + upcoming)\n"
    "        )\n"
    "        model_kwargs[\"cache_position\"] = torch.arange(\n"
    "            cached_length, cached_length + upcoming, device=cache_position.device\n"
    "        )\n"
    "        return model_kwargs\n"
    "\n"
    "    def prepare_inputs_for_generation(\n"
)

_UPDATE_MODEL_KWARGS_ANCHOR = (
    "            rope_deltas=self.rope_deltas,\n"
    "        )\n"
    "\n"
    "    def prepare_inputs_for_generation(\n"
)

HIPRUNE_QWEN_MULTI_IMAGE_CACHE_RESYNC = Patch(
    identifier="hiprune-qwen-multi-image-cache-resync-v1",
    target=Path("Qwen2_5_VL") / "qwen2_5_vl_HiPrune.py",
    original=_UPDATE_MODEL_KWARGS_ANCHOR,
    replacement=(
        "            rope_deltas=self.rope_deltas,\n"
        "        )\n"
        "\n"
        + UPDATE_MODEL_KWARGS_METHOD.format(
            identifier="hiprune-qwen-multi-image-cache-resync-v1"
        )
    ),
)

VISIONZIP_QWEN_MULTI_IMAGE_CACHE_RESYNC = Patch(
    identifier="visionzip-qwen-multi-image-cache-resync-v1",
    target=Path("Qwen2_5_VL") / "qwen2_5vl_visionzip.py",
    original=_UPDATE_MODEL_KWARGS_ANCHOR,
    replacement=(
        "            rope_deltas=self.rope_deltas,\n"
        "        )\n"
        "\n"
        + UPDATE_MODEL_KWARGS_METHOD.format(
            identifier="visionzip-qwen-multi-image-cache-resync-v1"
        )
    ),
)

PATCH_SETS = {
    "hiprune-qwen": (
        HIPRUNE_QWEN_MULTI_IMAGE_MASK,
        HIPRUNE_QWEN_MULTI_IMAGE_CACHE_POSITION,
        HIPRUNE_QWEN_MULTI_IMAGE_CACHE_RESYNC,
    ),
    "visionzip-qwen": (
        VISIONZIP_QWEN_MULTI_IMAGE_MASK,
        VISIONZIP_QWEN_MULTI_IMAGE_GATHER,
        VISIONZIP_QWEN_MULTI_IMAGE_CACHE_POSITION,
        VISIONZIP_QWEN_MULTI_IMAGE_CACHE_RESYNC,
    ),
}


def patch_source(source: str, patch: Patch) -> str | None:
    """Return the patched source, or None when the patch is already applied."""

    if patch.replacement in source:
        return None
    occurrences = source.count(patch.original)
    if occurrences != 1:
        raise SystemExit(
            f"Cannot apply {patch.identifier}: expected exactly one anchor in "
            f"{patch.target}, found {occurrences}. The pinned revision changed; "
            "review the patch before benchmarking."
        )
    return source.replace(patch.original, patch.replacement)


def write_atomically(path: Path, content: str) -> None:
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
        temporary.chmod(path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def apply_patch(root: Path, patch: Patch, verify_only: bool) -> None:
    path = root / patch.target
    if not path.is_file():
        raise SystemExit(f"Cannot apply {patch.identifier}: missing {path}")

    source = path.read_text(encoding="utf-8")
    patched = patch_source(source, patch)
    if patched is None:
        print(f"{patch.identifier} is already applied to {path}")
        return
    if verify_only:
        raise SystemExit(f"{patch.identifier} is not applied to {path}")

    write_atomically(path, patched)
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        write_atomically(path, source)
        raise SystemExit(
            f"Reverted {patch.identifier}: patched source does not compile: {exc}"
        )
    print(f"Applied {patch.identifier} to {path}")


def apply(root: Path, method: str, verify_only: bool) -> None:
    for patch in PATCH_SETS[method]:
        apply_patch(root, patch, verify_only)


def patch_id(method: str) -> str:
    return "+".join(patch.identifier for patch in PATCH_SETS[method])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(PATCH_SETS))
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Cloned pinned checkout root (the method's METHOD_ROOT).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Exit nonzero when a patch is absent instead of applying it.",
    )
    parser.add_argument(
        "--print-id",
        action="store_true",
        help="Print the patch identifiers recorded in run manifests and exit.",
    )
    args = parser.parse_args()
    if args.print_id:
        print(patch_id(args.method))
        return
    apply(args.root, args.method, args.verify)


if __name__ == "__main__":
    main()
