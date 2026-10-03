from __future__ import annotations

from collections.abc import Callable


# A stage's completed and total work units, independent of any display.
Progress = Callable[[str, int, int], None]


def report(progress: Progress | None, stage: str, completed: int, total: int = 1) -> None:
    if progress is not None:
        progress(stage, int(completed), int(total))
