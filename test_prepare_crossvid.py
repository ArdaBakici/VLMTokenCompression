import json
import tempfile
import unittest
from pathlib import Path

from crossvid_score import TASKS
from scripts.prepare_crossvid import (
    EXPECTED_TOTAL,
    missing_media,
    uav_references,
    validate,
    video_references,
)

# One example per task, sized so the fixture reaches the released total.
VIDEO_LIST_TASKS = ("BU", "NC", "CC", "PEA")
VIDEO_PAIR_TASKS = ("FSA", "CCQA")
VIDEO_SINGLE_TASKS = ("PI", "PSS")
UAV_TASKS = ("MSR", "MOC")


def example(task: str, index: int) -> dict:
    pair: dict = {"id": f"{task}-{index}"}
    if task in VIDEO_LIST_TASKS:
        pair["videos"] = [f"cook/{task}.mp4", f"behavior/{task}.mp4"]
    elif task in VIDEO_PAIR_TASKS:
        pair["video A"] = f"cook/{task}-a.mp4"
        pair["video B"] = f"cook/{task}-b.mp4"
    elif task in VIDEO_SINGLE_TASKS:
        pair["video"] = f"movie/{task}.mp4"
    else:
        pair["vid"] = f"{task}-class"
    return pair


def build_dataset(root: Path, complete: bool) -> None:
    """Write a fixture whose task counts add up to the released total."""

    qa_dir = root / "QA"
    qa_dir.mkdir(parents=True)
    remaining = EXPECTED_TOTAL - (len(TASKS) - 1)
    for position, task in enumerate(TASKS):
        count = remaining if position == 0 else 1
        pairs = [example(task, index) for index in range(count)]
        (qa_dir / f"{task}.json").write_text(json.dumps(pairs), encoding="utf-8")

    for task in TASKS:
        for reference in video_references(task, example(task, 0)):
            if not complete and reference.startswith("behavior/"):
                continue
            path = root / "videos" / reference
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"video")
        if task not in UAV_TASKS:
            continue
        for reference, is_directory in uav_references(example(task, 0), task):
            path = root / "uav" / reference
            if is_directory:
                path.mkdir(parents=True, exist_ok=True)
                (path / "000001.jpg").write_bytes(b"frame")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[]", encoding="utf-8")


class MediaReferenceTest(unittest.TestCase):
    def test_reads_the_official_media_fields(self):
        self.assertEqual(
            video_references("BU", {"id": "1", "videos": ["cook/a.mp4"]}),
            ["cook/a.mp4"],
        )
        self.assertEqual(
            video_references("FSA", {"id": "1", "video A": "a.mp4", "video B": "b.mp4"}),
            ["a.mp4", "b.mp4"],
        )
        self.assertEqual(video_references("PI", {"id": "1", "video": "m.mp4"}), ["m.mp4"])
        self.assertEqual(video_references("MSR", {"id": "1", "vid": "x"}), [])

    def test_uav_examples_need_both_views(self):
        references = uav_references({"id": "1", "vid": "x"}, "MOC")
        self.assertEqual(
            references,
            [
                ("bbox/1/x.json", False),
                ("frames/1/x-1", True),
                ("bbox/2/x.json", False),
                ("frames/2/x-2", True),
            ],
        )

    def test_rejects_annotations_without_media_fields(self):
        with self.assertRaises(SystemExit):
            video_references("BU", {"id": "1"})
        with self.assertRaises(SystemExit):
            uav_references({"id": "1"}, "MSR")


class ValidationTest(unittest.TestCase):
    def test_accepts_a_complete_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dataset(root, complete=True)
            marker = root / ".prepared"

            self.assertEqual(missing_media(root / "QA", root / "videos", root / "uav"), [])
            validate(root, marker, allow_restricted=False)
            self.assertTrue(marker.is_file())

    def test_reports_missing_license_restricted_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dataset(root, complete=False)
            marker = root / ".prepared"

            missing = missing_media(root / "QA", root / "videos", root / "uav")
            self.assertTrue(missing)
            self.assertTrue(all(path.startswith("videos/behavior/") for path in missing))

            with self.assertRaises(SystemExit):
                validate(root, marker, allow_restricted=False)
            # The redistributable part of the release is still intact.
            self.assertTrue(marker.is_file())
            validate(root, marker, allow_restricted=True)

    def test_rejects_an_unexpected_example_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dataset(root, complete=True)
            marker = root / ".prepared"
            marker.write_text("stale\n", encoding="utf-8")
            (root / "QA" / "BU.json").write_text("[]", encoding="utf-8")

            with self.assertRaises(SystemExit):
                validate(root, marker, allow_restricted=False)
            self.assertFalse(marker.exists())

    def test_rejects_media_that_escapes_the_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dataset(root, complete=True)
            (root / "QA" / "PI.json").write_text(
                json.dumps([{"id": "PI-0", "video": "../escape.mp4"}]), encoding="utf-8"
            )

            with self.assertRaises(SystemExit):
                missing_media(root / "QA", root / "videos", root / "uav")


if __name__ == "__main__":
    unittest.main()
