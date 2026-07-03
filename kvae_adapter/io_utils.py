from __future__ import annotations

import os
from pathlib import Path


def fadvise_dontneed(path: str | Path, *, sync: bool = False) -> None:
    """Best-effort request to drop a file's pages from the Linux page cache."""
    if not hasattr(os, "posix_fadvise"):
        return
    path = Path(path)
    if not path.exists() or path.is_dir():
        return
    flags = os.O_RDONLY
    fd = -1
    try:
        fd = os.open(path, flags)
        if sync:
            os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        return
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
