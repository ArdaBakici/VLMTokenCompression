import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_mmiu import missing_media, prepare, validate


class MmiuPreparationTest(unittest.TestCase):
    archive_name = "High-level-sub-semantic.zip"
    relative_image = (
        "High-level-sub-semantic/multiple_image_captioning/"
        "multiple_image_captioning_0_0.jpg"
    )

    def write_metadata(self, root: Path) -> None:
        table = pa.table({"input_image_path": [[f"./{self.relative_image}"]]})
        pq.write_table(table, root / "all.parquet")

    def write_archive(self, root: Path) -> None:
        with zipfile.ZipFile(root / self.archive_name, "w") as archive:
            archive.writestr(
                "multiple_image_captioning/multiple_image_captioning_0_0.jpg",
                b"image",
            )

    @patch("scripts.prepare_mmiu.ARCHIVES", (archive_name,))
    def test_extracts_archive_under_its_stem(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_metadata(root)
            self.write_archive(root)

            prepare(root)

            self.assertTrue((root / self.relative_image).is_file())
            self.assertEqual(missing_media(root), [])

    @patch("scripts.prepare_mmiu.ARCHIVES", (archive_name,))
    def test_migrates_previous_flat_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_metadata(root)
            self.write_archive(root)
            old_image = (
                root
                / "multiple_image_captioning"
                / "multiple_image_captioning_0_0.jpg"
            )
            old_image.parent.mkdir()
            old_image.write_bytes(b"image")

            prepare(root)

            self.assertFalse(old_image.exists())
            self.assertTrue((root / self.relative_image).is_file())

    def test_failed_validation_invalidates_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_metadata(root)
            marker = root / ".prepared"
            marker.touch()

            with self.assertRaises(SystemExit):
                validate(root, marker)

            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
