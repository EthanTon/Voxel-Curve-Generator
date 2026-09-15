from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass

from .cross_section import CrossSection


@dataclass(frozen=True, slots=True)
class SectionSpan:
    # The section delivered at every path point from path_start through
    # path_stop, inclusive, the same as PathFrame.
    path_start: int
    path_stop: int
    section: CrossSection

    def __post_init__(self) -> None:
        if self.path_start < 0:
            raise ValueError(f"path_start ({self.path_start}) is negative")
        if self.path_stop < self.path_start:
            raise ValueError(
                f"path_stop ({self.path_stop}) precedes path_start ({self.path_start})"
            )

    def contains(self, path_index: int) -> bool:
        return self.path_start <= path_index <= self.path_stop


class SectionSchedule:
    # Section spans along a path. Spans may leave gaps, where nothing is
    # delivered, but may not overlap.

    def __init__(self, spans: Iterable[SectionSpan], path_length: int) -> None:
        self._spans = tuple(sorted(spans, key=lambda span: span.path_start))
        for before, after in zip(self._spans, self._spans[1:]):
            if after.path_start <= before.path_stop:
                raise ValueError(
                    f"Section spans overlap: {before.path_start}..{before.path_stop} "
                    f"and {after.path_start}..{after.path_stop}"
                )
        if self._spans and self._spans[-1].path_stop >= path_length:
            raise ValueError(
                f"Section span ends at {self._spans[-1].path_stop}, "
                f"past the last path index ({path_length - 1})"
            )
        self._starts = tuple(span.path_start for span in self._spans)

    @property
    def spans(self) -> tuple[SectionSpan, ...]:
        return self._spans

    def at(self, path_index: int) -> CrossSection | None:
        # Random access, for previews and tests.
        i = bisect_right(self._starts, path_index) - 1
        if i >= 0 and self._spans[i].contains(path_index):
            return self._spans[i].section
        return None
