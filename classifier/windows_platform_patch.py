from __future__ import annotations

import platform
import sys
from typing import Any


def patch_windows_platform_wmi() -> None:
    """Avoid Python platform WMI calls that can hang torch/ultralytics imports on Windows."""
    if sys.platform != "win32":
        return
    uname_result = getattr(platform, "uname_result", None)
    if uname_result is None:
        return

    def fixed_uname() -> Any:
        values = ("Windows", "Windows-PC", "10", "10", "AMD64", "AMD64")
        try:
            return uname_result(*values)
        except TypeError:
            return uname_result(values)

    platform.system = lambda: "Windows"  # type: ignore[assignment]
    platform.machine = lambda: "AMD64"  # type: ignore[assignment]
    platform.processor = lambda: "AMD64"  # type: ignore[assignment]
    platform.release = lambda: "10"  # type: ignore[assignment]
    platform.version = lambda: "10"  # type: ignore[assignment]
    platform.win32_ver = lambda: ("10", "", "", "Multiprocessor Free")  # type: ignore[assignment]
    platform.uname = fixed_uname  # type: ignore[assignment]
