"""Let vLLM's MoE extension load on hosts below the wheel's glibc baseline.

The precompiled manylinux_2_31 wheel builds `vllm/_moe_C.abi3.so` against a newer
glibc than some compute nodes provide, and the whole difference is one symbol:
`log2@GLIBC_2.29`. Every other version the extension imports is glibc 2.14 or
older. glibc has exported `log2@GLIBC_2.2.5` since 2.2.5 and added the 2.29 entry
as a faster implementation of the same function, so an older host can satisfy the
call as long as the binary stops insisting on the newer version.

Two independent things enforce that insistence, and both have to go:

  1. `.gnu.version`, which binds the `log2` import to version `GLIBC_2.29`. This
     is what `patchelf --clear-symbol-version` rewrites.
  2. `.gnu.version_r`, the table of versions each needed library must provide.
     The loader walks it in `_dl_check_map_versions` before any symbol is bound,
     and a missing non-weak entry is fatal: it is what prints
     "version `GLIBC_2.29' not found (required by ...)". patchelf leaves this
     alone, so clearing the symbol version by itself changes nothing.

This module rewrites both, in place and without resizing anything: the symbol's
version index becomes `VER_NDX_GLOBAL`, and the version requirement is marked
`VER_FLG_WEAK` so the loader reports it as absent instead of refusing to load.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

SYMBOL = "log2"
VERSION = "GLIBC_2.29"
LIBRARY = "libm.so.6"

VER_NDX_GLOBAL = 1
VER_FLG_WEAK = 0x2

SHT_DYNSYM = 11
SHT_GNU_VERSYM = 0x6FFFFFFF
SHT_GNU_VERNEED = 0x6FFFFFFE


@dataclass(frozen=True)
class Section:
    name: str
    type: int
    offset: int
    size: int
    link: int
    entsize: int


def read_sections(data: bytes) -> list[Section]:
    if data[:4] != b"\x7fELF":
        raise SystemExit("Not an ELF file")
    if data[4] != 2:
        raise SystemExit("Only 64-bit ELF files are supported")

    sh_offset, = struct.unpack_from("<Q", data, 0x28)
    sh_entry_size, sh_count, sh_strndx = struct.unpack_from("<HHH", data, 0x3A)

    raw = []
    for index in range(sh_count):
        base = sh_offset + index * sh_entry_size
        name, type_, _flags, _addr, offset, size, link, _info, _align, entsize = (
            struct.unpack_from("<IIQQQQIIQQ", data, base)
        )
        raw.append((name, type_, offset, size, link, entsize))

    strtab_offset = raw[sh_strndx][2]

    def name_at(position: int) -> str:
        end = data.index(b"\0", strtab_offset + position)
        return data[strtab_offset + position : end].decode()

    return [
        Section(name_at(name), type_, offset, size, link, entsize)
        for name, type_, offset, size, link, entsize in raw
    ]


def find(sections: list[Section], type_: int) -> Section:
    for section in sections:
        if section.type == type_:
            return section
    raise SystemExit(f"The extension has no section of type {type_:#x}")


def string_at(data: bytes, table: Section, position: int) -> str:
    start = table.offset + position
    return data[start : data.index(b"\0", start)].decode()


def plan_edits(data: bytes) -> list[tuple[int, bytes, str]]:
    """Return (offset, replacement, description) edits, empty when already done."""

    sections = read_sections(data)
    verneed = find(sections, SHT_GNU_VERNEED)
    versym = find(sections, SHT_GNU_VERSYM)
    dynsym = find(sections, SHT_DYNSYM)
    strtab = sections[verneed.link]

    edits: list[tuple[int, bytes, str]] = []
    version_index = None

    offset = verneed.offset
    while True:
        _version, count, file_name, aux, next_entry = struct.unpack_from(
            "<HHIII", data, offset
        )
        library = string_at(data, strtab, file_name)
        aux_offset = offset + aux
        for _ in range(count):
            _hash, flags, other, name, next_aux = struct.unpack_from(
                "<IHHII", data, aux_offset
            )
            if library == LIBRARY and string_at(data, strtab, name) == VERSION:
                version_index = other
                if not flags & VER_FLG_WEAK:
                    edits.append(
                        (
                            aux_offset + 4,
                            struct.pack("<H", flags | VER_FLG_WEAK),
                            f"mark {library} {VERSION} requirement weak",
                        )
                    )
            if not next_aux:
                break
            aux_offset += next_aux
        if not next_entry:
            break
        offset += next_entry

    if version_index is None:
        return []

    symbol_strtab = sections[dynsym.link]
    count = dynsym.size // dynsym.entsize
    for index in range(count):
        name, = struct.unpack_from("<I", data, dynsym.offset + index * dynsym.entsize)
        if string_at(data, symbol_strtab, name) != SYMBOL:
            continue
        entry = versym.offset + index * 2
        current, = struct.unpack_from("<H", data, entry)
        if current == version_index:
            edits.append(
                (
                    entry,
                    struct.pack("<H", VER_NDX_GLOBAL),
                    f"bind {SYMBOL} to the base version",
                )
            )
    return edits


def rebind(path: Path, verify_only: bool) -> None:
    if not path.is_file():
        raise SystemExit(f"No such extension: {path}")
    data = path.read_bytes()
    edits = plan_edits(data)
    if not edits:
        print(f"{path} already loads without {VERSION}")
        return
    if verify_only:
        raise SystemExit(f"{path} still requires {VERSION}")

    with path.open("r+b") as handle:
        for offset, replacement, description in edits:
            handle.seek(offset)
            handle.write(replacement)
            print(f"  {description}")
    print(f"Rebound {SYMBOL} in {path}")


def locate() -> Path:
    import vllm

    extensions = sorted(Path(vllm.__file__).parent.glob("_moe_C*.so"))
    if not extensions:
        raise SystemExit("The installed vLLM has no MoE extension")
    return extensions[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "extension",
        nargs="?",
        type=Path,
        help="Path to _moe_C.abi3.so. Defaults to the installed vLLM's copy.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Exit nonzero when the rebinding is absent instead of applying it.",
    )
    args = parser.parse_args()
    rebind(args.extension or locate(), args.verify)


if __name__ == "__main__":
    main()
