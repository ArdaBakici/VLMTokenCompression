import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.rebind_moe_glibc import (
    SYMBOL,
    VER_FLG_WEAK,
    VER_NDX_GLOBAL,
    VERSION,
    plan_edits,
    read_sections,
    rebind,
)

# A real extension is 257 MB, so build the smallest ELF that carries the four
# sections the rebinding reads. Written with the same layout the loader uses:
# a version requirement on libm.so.6 and one versioned undefined symbol.
LIBRARY = b"libm.so.6\0"


def build_extension(path: Path, version: str = VERSION, symbol: str = SYMBOL) -> None:
    strings = b"\0" + LIBRARY + version.encode() + b"\0" + symbol.encode() + b"\0"
    version_offset = 1 + len(LIBRARY)
    symbol_offset = version_offset + len(version) + 1
    section_names = b"\0.dynsym\0.dynstr\0.gnu.version\0.gnu.version_r\0.shstrtab\0"

    # Two dynsym entries: the reserved null symbol and the versioned import.
    dynsym = struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0)
    dynsym += struct.pack("<IBBHQQ", symbol_offset, 0x12, 0, 0, 0, 0)
    versym = struct.pack("<HH", 0, 16)
    verneed = struct.pack("<HHIII", 1, 1, 1, 16, 0)
    verneed += struct.pack("<IHHII", 0, 0, 16, version_offset, 0)

    parts, offset, layout = [], 0x40, {}
    for name, blob in (
        ("dynsym", dynsym),
        ("dynstr", strings),
        ("versym", versym),
        ("verneed", verneed),
        ("shstrtab", section_names),
    ):
        layout[name] = (offset, len(blob))
        parts.append(blob)
        offset += len(blob)

    headers = b""
    # Section order is dynsym, dynstr, versym, verneed, shstrtab, so sh_link
    # points at index 1 (.dynstr) for the two tables that name strings, and at
    # index 0 (.dynsym) for the version-symbol table.
    # sh_info carries the verneed entry count, which readelf reads to decide how
    # many requirements to print.
    for name, name_offset, type_, link, info, entsize in (
        ("dynsym", 1, 11, 1, 0, 24),
        ("dynstr", 9, 3, 0, 0, 0),
        ("versym", 17, 0x6FFFFFFF, 0, 0, 2),
        ("verneed", 30, 0x6FFFFFFE, 1, 1, 0),
        ("shstrtab", 45, 3, 0, 0, 0),
    ):
        start, size = layout[name]
        headers += struct.pack(
            "<IIQQQQIIQQ", name_offset, type_, 0, 0, start, size, link, info, 1, entsize
        )

    header = bytearray(0x40)
    header[0:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    struct.pack_into("<Q", header, 0x28, offset)
    struct.pack_into("<HHH", header, 0x3A, 64, 5, 4)
    path.write_bytes(bytes(header) + b"".join(parts) + headers)


class ReadSectionsTest(unittest.TestCase):
    def test_rejects_non_elf_and_32_bit(self):
        with self.assertRaises(SystemExit):
            read_sections(b"not an elf file")
        with self.assertRaises(SystemExit):
            read_sections(b"\x7fELF\x01" + bytes(64))


class PlanEditsTest(unittest.TestCase):
    def test_plans_both_edits_then_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "_moe_C.abi3.so"
            build_extension(path)

            edits = plan_edits(path.read_bytes())
            self.assertEqual(len(edits), 2)
            self.assertEqual(
                [description for _, _, description in edits],
                [
                    "mark libm.so.6 GLIBC_2.29 requirement weak",
                    "bind log2 to the base version",
                ],
            )

            rebind(path, verify_only=False)
            self.assertEqual(plan_edits(path.read_bytes()), [])

    def test_ignores_an_extension_without_the_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "_moe_C.abi3.so"
            build_extension(path, version="GLIBC_2.14")
            self.assertEqual(plan_edits(path.read_bytes()), [])


class RebindTest(unittest.TestCase):
    def test_sets_weak_flag_and_global_version_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "_moe_C.abi3.so"
            build_extension(path)
            before = path.read_bytes()
            rebind(path, verify_only=False)
            after = path.read_bytes()

            self.assertEqual(len(before), len(after))
            differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
            self.assertEqual(len(differing), 2)

            sections = read_sections(after)
            verneed = next(s for s in sections if s.type == 0x6FFFFFFE)
            versym = next(s for s in sections if s.type == 0x6FFFFFFF)
            _hash, flags, _other, _name, _next = struct.unpack_from(
                "<IHHII", after, verneed.offset + 16
            )
            self.assertTrue(flags & VER_FLG_WEAK)
            index, = struct.unpack_from("<H", after, versym.offset + 2)
            self.assertEqual(index, VER_NDX_GLOBAL)

    def test_verify_reports_both_states(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "_moe_C.abi3.so"
            build_extension(path)
            with self.assertRaises(SystemExit):
                rebind(path, verify_only=True)
            rebind(path, verify_only=False)
            rebind(path, verify_only=True)

    def test_requires_the_extension_to_exist(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit):
            rebind(Path(directory) / "absent.so", verify_only=False)


@unittest.skipUnless(shutil.which("readelf"), "readelf is unavailable")
class ReadelfAgreementTest(unittest.TestCase):
    def test_readelf_sees_the_weak_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "_moe_C.abi3.so"
            build_extension(path)
            rebind(path, verify_only=False)
            output = subprocess.run(
                ["readelf", "-V", str(path)], capture_output=True, text=True, check=True
            ).stdout
            self.assertIn("Flags: WEAK", output)


if __name__ == "__main__":
    sys.exit(unittest.main())
