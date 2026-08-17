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

PATCH_SETS = {
    "hiprune-qwen": (HIPRUNE_QWEN_MULTI_IMAGE_MASK,),
    "visionzip-qwen": (
        VISIONZIP_QWEN_MULTI_IMAGE_MASK,
        VISIONZIP_QWEN_MULTI_IMAGE_GATHER,
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
