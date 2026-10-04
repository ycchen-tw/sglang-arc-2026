"""arc-forge MTP5-DFlash drafter (experimental-dflash d01/l01): swaps the native MTP draft worker of EAGLEWorkerV2.

``ARC_DFLASH_DRAFTER=<export dir>`` turns it on. The value is the drafter's export directory, so any non-empty
value must name an existing directory: ``0``/``false`` or a stale path raise instead of silently enabling it.
"""

from __future__ import annotations

import os


_checked: set[str] = set()  # export directories already found to exist


def drafter_dir() -> str | None:
    """The DFlash drafter export directory, or None when DFlash is off.

    The scheduler asks every decode cycle (overlap_utils), and the export usually sits on a network volume where one
    stat costs ~1 ms of host time the GPU waits for: each path is checked once per process."""
    path = os.environ.get("ARC_DFLASH_DRAFTER")
    if not path:
        return None
    if path not in _checked:
        if not os.path.isdir(path):
            raise RuntimeError(f"ARC_DFLASH_DRAFTER={path!r} is not a drafter export directory")
        _checked.add(path)
    return path


def enabled() -> bool:
    return drafter_dir() is not None
