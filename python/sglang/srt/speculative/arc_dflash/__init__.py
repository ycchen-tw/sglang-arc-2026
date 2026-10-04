"""arc-forge MTP5-DFlash drafter (experimental-dflash d01/l01): swaps the native MTP draft worker of EAGLEWorkerV2.

``ARC_DFLASH_DRAFTER=<export dir>`` turns it on. The value is the drafter's export directory, so any non-empty
value must name an existing directory: ``0``/``false`` or a stale path raise instead of silently enabling it.
"""

from __future__ import annotations

import os


def drafter_dir() -> str | None:
    """The DFlash drafter export directory, or None when DFlash is off."""
    path = os.environ.get("ARC_DFLASH_DRAFTER")
    if not path:
        return None
    if not os.path.isdir(path):
        raise RuntimeError(f"ARC_DFLASH_DRAFTER={path!r} is not a drafter export directory")
    return path


def enabled() -> bool:
    return drafter_dir() is not None
