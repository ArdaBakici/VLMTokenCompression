"""Validate the pinned CrossVid annotation and media layout."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Any

from crossvid_score import TASKS, index_by_id, read_list

EXPECTED_TOTAL = 9015
VIDEO_LIST_TASKS = {"BU", "NC", "CC", "PEA"}
VIDEO_PAIR_TASKS = {"FSA", "CCQA"}
VIDEO_SINGLE_TASKS = {"PI", "PSS"}
UAV_TASKS = {"MSR", "MOC"}

# Charades and Animal Kingdom source videos carry licenses that forbid
# redistribution, so the pinned release ships assembly, cook, and movie genres
# only. The behavior genre must be obtained from the original repositories.
RESTRICTED_GENRE = "behavior"
RESTRICTED_HELP = (
    "Charades and Animal Kingdom forbid redistribution, so the pinned CrossVid "
    "release omits the behavior genre. Obtain those videos from their original "
    "repositories and merge them into videos/behavior, keeping the file names "
    "referenced by the annotations."
)


def relative_path(root: Path, value: str) -> Path:
    """Resolve a media reference below root, rejecting escapes."""

    relative = value
    while relative.startswith("./"):
        relative = relative[2:]
    if PurePosixPath(relative).is_absolute():
        raise SystemExit(f"Absolute media reference is not allowed: {value}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"Media reference escapes {root}: {value}") from exc
    return path


def require(pair: dict[str, Any], key: str, task: str) -> Any:
    if key not in pair:
        raise SystemExit(
            f"CrossVid {task} record {pair.get('id', '<no id>')!r} has no {key!r} "
            "field; the annotations do not match the pinned revision"
        )
    return pair[key]


def video_references(task: str, pair: dict[str, Any]) -> list[str]:
    """Return the video paths one example reads, relative to the video root."""

    if task in VIDEO_LIST_TASKS:
        return [str(video) for video in require(pair, "videos", task)]
    if task in VIDEO_PAIR_TASKS:
        return [str(require(pair, "video A", task)), str(require(pair, "video B", task))]
    if task in VIDEO_SINGLE_TASKS:
        return [str(require(pair, "video", task))]
    return []


def uav_references(pair: dict[str, Any], task: str) -> list[tuple[str, bool]]:
    """Return (path, is_directory) UAV references for one MSR or MOC example."""

    class_id = require(pair, "vid", task)
    references = []
    for view in (1, 2):
        references.append((f"bbox/{view}/{class_id}.json", False))
        references.append((f"frames/{view}/{class_id}-{view}", True))
    return references


def present(path: Path, is_directory: bool) -> bool:
    if not is_directory:
        return path.is_file()
    return path.is_dir() and any(path.iterdir())


def missing_media(qa_dir: Path, video_root: Path, uav_root: Path) -> list[str]:
    """Return every referenced media path that is absent, relative to the root."""

    missing = set()
    for task in TASKS:
        qa_path = qa_dir / f"{task}.json"
        for pair in index_by_id(read_list(qa_path), qa_path).values():
            for reference in video_references(task, pair):
                if not relative_path(video_root, reference).is_file():
                    missing.add(f"videos/{reference}")
            if task not in UAV_TASKS:
                continue
            for reference, is_directory in uav_references(pair, task):
                if not present(relative_path(uav_root, reference), is_directory):
                    missing.add(f"uav/{reference}")
    return sorted(missing)


def count_examples(qa_dir: Path) -> dict[str, int]:
    return {
        task: len(index_by_id(read_list(qa_dir / f"{task}.json"), qa_dir / f"{task}.json"))
        for task in TASKS
    }


def report(missing: list[str], label: str, help_text: str) -> None:
    preview = ", ".join(missing[:3])
    print(f"{label}: {len(missing)} referenced paths are absent; first: {preview}")
    print(help_text)


def list_missing(root: Path, output: Path | None) -> None:
    """Print every absent media path, one per line, for fetching or diffing."""

    missing = missing_media(root / "QA", root / "videos", root / "uav")
    text = "".join(f"{path}\n" for path in missing)
    if output is None:
        print(text, end="")
    else:
        output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(missing)} absent paths to {output}")


def validate(root: Path, marker: Path, allow_restricted: bool) -> None:
    qa_dir = root / "QA"
    counts = count_examples(qa_dir)
    total = sum(counts.values())
    if total != EXPECTED_TOTAL:
        marker.unlink(missing_ok=True)
        raise SystemExit(
            f"CrossVid annotations hold {total} examples, expected {EXPECTED_TOTAL}. "
            f"Inspect them with: crossvid-score inspect --qa-dir {qa_dir}"
        )
    print(f"CrossVid annotations validated: {total} examples across {len(TASKS)} tasks.")

    prefix = f"videos/{RESTRICTED_GENRE}/"
    missing = missing_media(qa_dir, root / "videos", root / "uav")
    restricted = [path for path in missing if path.startswith(prefix)]
    other = [path for path in missing if not path.startswith(prefix)]

    if other:
        marker.unlink(missing_ok=True)
        report(
            other,
            "CrossVid media validation failed",
            f"Redownload the pinned release into {root}.",
        )
        raise SystemExit(f"CrossVid media validation failed: {len(other)} files missing")

    # The marker records that the redistributable part of the pinned release is
    # intact. The behavior genre is never part of that download.
    marker.write_text("validated\n", encoding="utf-8")

    if restricted:
        report(restricted, "License-restricted media are absent", RESTRICTED_HELP)
        if not allow_restricted:
            raise SystemExit(
                "Refusing to benchmark an incomplete CrossVid: every example that "
                "reads these videos would be recorded as a failure and would still "
                "count in the denominator of the reported score. Set "
                "CROSSVID_ALLOW_MISSING_BEHAVIOR=1 to run the remaining examples "
                "anyway, and report the reduced coverage."
            )
        print(
            "WARNING: continuing without the behavior genre. Examples that read "
            "these videos will fail and will still count in the reported score. "
            "Report this as reduced coverage rather than a CrossVid score."
        )
        return

    print("CrossVid media validation passed: all referenced files are present.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument(
        "--allow-missing-behavior",
        action="store_true",
        help="Benchmark the available genres when the behavior videos are absent.",
    )
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="Print every absent media path instead of validating.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="With --list-missing, write the paths to this file instead of stdout.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.list_missing:
        list_missing(root, args.output)
        return
    if args.marker is None:
        raise SystemExit("--marker is required unless --list-missing is given")
    validate(root, args.marker, args.allow_missing_behavior)


if __name__ == "__main__":
    main()
