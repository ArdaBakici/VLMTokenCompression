"""Apply the local vLLM patches that the image-pruning profile depends on.

`sequential-image-encoding-v1` makes the vision tower encode images one at a
time. vLLM charges an image against the multimodal encoder compute budget using
the number of tokens it contributes to the prompt, which pruning reduces, while
the vision tower still runs on every unpruned patch. The scheduler therefore
packs far more images into one vision-tower forward than memory profiling
assumed, which exhausts device memory. vLLM already avoids this for Efficient
Video Sampling, but gates that path to the video modality; this patch extends
the gate to images.

`moe-image-pruning-rate-v1` lets the pruning profile serve the Qwen3-VL MoE
checkpoints. `Qwen3VLMoeForConditionalGeneration.__init__` calls `super()` on the
grandparent class, skipping the dense `__init__` that assigns
`image_pruning_rate`, yet it inherits every image path that reads that attribute.
Both classes share one multimodal processor, so the prompt placeholders are
already shortened by the pruning rate before the model raises `AttributeError` on
the first image.

Each patch is applied to the installed pinned build, is idempotent, and refuses
to run when the pinned source no longer matches what it expects.
"""

from __future__ import annotations

import argparse
import importlib.util
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


SEQUENTIAL_IMAGE_ENCODING = Patch(
    identifier="sequential-image-encoding-v1",
    target=Path("v1/worker/gpu_model_runner.py"),
    original="""\
                and modality == "video"
                and num_items > 1
""",
    replacement="""\
                # Local patch: sequential-image-encoding-v1.
                # The encoder compute budget is spent in post-pruning tokens,
                # so attention-based image pruning lets the scheduler place far
                # more images in one vision-tower forward than memory profiling
                # assumed. Images need the same sequential encoding that video
                # pruning already uses.
                and modality in ("video", "image")
                and num_items > 1
""",
)

MOE_IMAGE_PRUNING_RATE = Patch(
    identifier="moe-image-pruning-rate-v1",
    target=Path("model_executor/models/qwen3_vl_moe.py"),
    original="""\
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
""",
    replacement="""\
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        # Local patch: moe-image-pruning-rate-v1.
        # This __init__ skips Qwen3VLForConditionalGeneration.__init__, which is
        # where the dense checkpoint assigns the attribute that every inherited
        # image path reads.
        self.image_pruning_rate = multimodal_config.image_pruning_rate
        self.is_multimodal_pruning_enabled = (
""",
)

PATCHES = (SEQUENTIAL_IMAGE_ENCODING, MOE_IMAGE_PRUNING_RATE)
PATCH_ID = "+".join(patch.identifier for patch in PATCHES)


def vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vLLM is not importable from this interpreter")
    return Path(next(iter(spec.submodule_search_locations)))


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


def apply(root: Path, verify_only: bool) -> None:
    for patch in PATCHES:
        apply_patch(root, patch, verify_only)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-root",
        type=Path,
        help="Installed vllm package directory. Defaults to the importable one.",
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
        print(PATCH_ID)
        return
    apply(args.vllm_root or vllm_root(), args.verify)


if __name__ == "__main__":
    main()
