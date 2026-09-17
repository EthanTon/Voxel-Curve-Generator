from __future__ import annotations

from collections.abc import Iterable

from ...Core.CrossSection.schedule import SectionSchedule, SectionSpan
from ...Core.Path.path import Path
from .boundaries import Orientation, orthogonal, path_bounds
from .frames import (
    build_frames,
    build_section_frames,
    elevation_breaks,
    elevation_from_frames,
    span_stations,
)
from .reach import lateral_reach


class PathGeometry:
    def __init__(
        self,
        points,
        gradients,
        section_spans: Iterable[SectionSpan] = (),
        *,
        padding_in: int = 0,
        padding_out: int = 0,
        orientation: Orientation = orthogonal,
    ):
        # Path points provided
        self.path = Path(
            points=points,
            gradients=gradients,
            lead_in=padding_in,
            lead_out=padding_out,
        )

        # Sections of the path provided
        self.sections = SectionSchedule(section_spans, len(self.path.path_points))

        # How far the sections extend either side of the path, at any turn
        self.reach = lateral_reach(span.section for span in self.sections.spans)

        # One frame per level, split wherever the path proper changes height
        start_y = self.path.path_points[self.path.first_index].position.y
        self.elevation_frames = build_frames(
            self.path,
            start_y,
            elevation_breaks(self.path),
            orientation=orientation,
            reach=self.reach,
        )

        # Frames split wherever the section or the level changes
        stations = sorted(
            {*span_stations(self.sections, self.path)}
            | {frame.path_start for frame in self.elevation_frames[1:]}
        )
        self.section_frames = build_section_frames(
            self.path,
            self.elevation_frames,
            stations,
            orientation=orientation,
            reach=self.reach,
        )

        # The frames' levels, looked up by column
        self.elevation = elevation_from_frames(self.elevation_frames)

        # The columns the path proper owns
        self.bounds = path_bounds(self.path, orientation)
