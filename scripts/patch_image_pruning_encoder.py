"""Make the image-pruning profile encode images sequentially in the vision tower.

vLLM charges an image against the multimodal encoder compute budget using the
number of tokens the image contributes to the prompt. The attention-based image
pruning of vLLM PR #38888 shrinks that number by the pruning rate, but the
vision tower still runs on every unpruned patch. The scheduler therefore packs
far more images into one vision-tower forward than memory profiling assumed,
which exhausts device memory. vLLM already avoids this for Efficient Video
Sampling by encoding pruned media one item at a time, but that path is gated to
the video modality. This patch extends the same gate to images.

The patch is applied to the installed pinned vLLM build, is idempotent, and
refuses to run when the pinned source no longer matches what it expects.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import py_compile
import tempfile
from pathlib import Path

PATCH_ID = "sequential-image-encoding-v1"
TARGET = Path("v1/worker/gpu_model_runner.py")

ORIGINAL = """\
                and modality == "video"
                and num_items > 1
"""

REPLACEMENT = f"""\
                # Local patch: {PATCH_ID}.
                # The encoder compute budget is spent in post-pruning tokens,
                # so attention-based image pruning lets the scheduler place far
                # more images in one vision-tower forward than memory profiling
                # assumed. Images need the same sequential encoding that video
                # pruning already uses.
                and modality in ("video", "image")
                and num_items > 1
"""


def vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vLLM is not importable from this interpreter")
    return Path(next(iter(spec.submodule_search_locations)))


def patch_source(source: str) -> str | None:
    """Return the patched source, or None when the patch is already applied."""

    if REPLACEMENT in source:
        return None
    occurrences = source.count(ORIGINAL)
    if occurrences != 1:
        raise SystemExit(
            f"Cannot apply {PATCH_ID}: expected exactly one sequential encoder "
            f"guard in the pinned vLLM source, found {occurrences}. The pinned "
            "revision changed; review the patch before benchmarking."
        )
    return source.replace(ORIGINAL, REPLACEMENT)


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


def apply(root: Path, verify_only: bool) -> None:
    path = root / TARGET
    if not path.is_file():
        raise SystemExit(f"Cannot apply {PATCH_ID}: missing {path}")

    source = path.read_text(encoding="utf-8")
    patched = patch_source(source)
    if patched is None:
        print(f"{PATCH_ID} is already applied to {path}")
        return
    if verify_only:
        raise SystemExit(f"{PATCH_ID} is not applied to {path}")

    write_atomically(path, patched)
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        write_atomically(path, source)
        raise SystemExit(f"Reverted {PATCH_ID}: patched source does not compile: {exc}")
    print(f"Applied {PATCH_ID} to {path}")


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
        help="Exit nonzero when the patch is absent instead of applying it.",
    )
    parser.add_argument(
        "--print-id",
        action="store_true",
        help="Print the patch identifier recorded in run manifests and exit.",
    )
    args = parser.parse_args()
    if args.print_id:
        print(PATCH_ID)
        return
    apply(args.vllm_root or vllm_root(), args.verify)


if __name__ == "__main__":
    main()
