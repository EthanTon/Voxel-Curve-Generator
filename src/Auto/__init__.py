from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .selector import Selector, Settings

__all__ = ["Selector", "Settings", "auto_assignments"]


def auto_assignments(build_data: Mapping, base: Path) -> list[dict]:
    selector = Selector(Settings.read(build_data, base), build_data)
    if selector.settings.output is not None:
        selector.write()
    return selector.assignments()
