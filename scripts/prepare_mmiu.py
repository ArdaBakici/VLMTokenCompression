"""Prepare and validate the pinned MMIU media layout."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVES = (
    "2D-spatial.zip",
    "3D-spatial.zip",
    "Continuous-temporal.zip",
    "Discrete-temporal.zip",
    "High-level-obj-semantic.zip",
    "High-level-sub-semantic.zip",
    "Low-level-semantic.zip",
)


def stored_path(root: Path, value: str) -> Path:
    relative = value
    while relative.startswith("./"):
        relative = relative[2:]
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def missing_media(root: Path) -> list[str]:
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise SystemExit("pyarrow is missing; synchronize the locked environment") from exc

    parquet_path = root / "all.parquet"
    if not parquet_path.is_file():
        raise SystemExit(f"Missing MMIU metadata: {parquet_path}")
    rows = parquet.read_table(
        parquet_path, columns=["input_image_path"]
    ).column("input_image_path")
    return [
        value
        for values in rows.to_pylist()
        for value in values
        if not stored_path(root, value).is_file()
    ]


def migrate_flat_layout(root: Path, archive: Path) -> int:
    destination = root / archive.stem
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        task_directories = {
            parts[0]
            for name in handle.namelist()
            if len(parts := PurePosixPath(name).parts) > 1
        }

    migrated = 0
    for task in sorted(task_directories):
        old_path = root / task
        new_path = destination / task
        if old_path.is_dir() and not new_path.exists():
            shutil.move(str(old_path), str(new_path))
            migrated += 1
    return migrated


def extract_archives(root: Path, archives: list[Path]) -> None:
    for index, archive in enumerate(archives, 1):
        destination = root / archive.stem
        destination.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(archives)}] Extracting {archive.name} into "
            f"{destination.name}/",
            flush=True,
        )
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)


def prepare(root: Path) -> None:
    archives = [root / name for name in ARCHIVES]
    absent = [path.name for path in archives if not path.is_file()]
    if absent:
        raise SystemExit(f"Missing MMIU archives: {', '.join(absent)}")
    invalid = [path.name for path in archives if not zipfile.is_zipfile(path)]
    if invalid:
        raise SystemExit(f"Invalid or incomplete MMIU archives: {', '.join(invalid)}")

    migrated = sum(migrate_flat_layout(root, archive) for archive in archives)
    if migrated:
        print(f"Migrated {migrated} task directories from the old flat layout.")

    missing = missing_media(root)
    if missing:
        print(f"Media layout is missing {len(missing)} files; extracting archives.")
        extract_archives(root, archives)


def validate(root: Path, marker: Path) -> None:
    missing = missing_media(root)
    if missing:
        marker.unlink(missing_ok=True)
        raise SystemExit(
            f"MMIU media validation failed: {len(missing)} files are missing; "
            f"first missing path: {missing[0]}"
        )
    print("MMIU media validation passed: all referenced files are present.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--marker", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.prepare:
        prepare(root)
    validate(root, args.marker)
    args.marker.write_text("validated\n", encoding="utf-8")


if __name__ == "__main__":
    main()
