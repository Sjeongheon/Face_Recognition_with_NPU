"""Safely copy only lfw.bin out of the notebook's WebFace ZIP archive.

No generic extractall: archives may contain unrelated paths. The .bin uses
pickle, so load it only when obtained from a trusted source.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import stat
from tempfile import NamedTemporaryFile
from zipfile import ZipFile


def prepare_lfw_bin(archive: str | Path, output: str | Path, *, overwrite: bool = False) -> Path:
    target = Path(output)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Already exists: {target} (use overwrite=True explicitly)")
    with ZipFile(archive) as zf:
        matches = [entry for entry in zf.infolist()
                   if not entry.is_dir() and Path(entry.filename).name.lower() == "lfw.bin"]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one lfw.bin in {archive}; found {len(matches)}")
        entry = matches[0]
        if stat.S_IFMT(entry.external_attr >> 16) == stat.S_IFLNK:
            raise ValueError("lfw.bin entry must not be a symlink")
        if entry.file_size > 2 * 1024 ** 3:
            raise ValueError("lfw.bin exceeds the maximum allowed size")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with NamedTemporaryFile(dir=target.parent, prefix=".lfw-", delete=False) as tmp:
                temp = Path(tmp.name)
                with zf.open(entry) as source:
                    shutil.copyfileobj(source, tmp)
            if target.exists() and not overwrite:
                raise FileExistsError(target)
            temp.replace(target)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    return target
