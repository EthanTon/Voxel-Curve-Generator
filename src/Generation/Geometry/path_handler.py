from __future__ import annotations

from collections.abc import Iterable

from ...Core.CrossSection.schedule import SectionSchedule, SectionSpan
from ...Core.Path.path import Path
from .boundaries import Orientation, orthogonal, path_bounds
from .padding import padded_path
from .frames import (
    build_frames,
    build_section_frames,
    elevation_breaks,
    elevation_from_frames,
    span_stations,
)
from .reach import lateral_reach, required_lead


class PathGeometry:
    def __init__(
        self,
        points,
        gradients,
        section_spans: Iterable[SectionSpan] = (),
        *,
        padding: int | None = None,
        orientation: Orientation = orthogonal,
    ):
        # The points describe the path proper. How far past them it has to be
        # padded is not authored with them: it is what the sections pasted along
        # it reach, which only generation knows. `padding` overrides that count.
        spans = tuple(section_spans)
        sections = tuple(span.section for span in spans)
        authored = Path(points=points, gradients=gradients)

        self.padding = (
            required_lead(authored, sections) if padding is None else padding
        )

        # Path points provided, extended into the lead each end needs
        self.path = padded_path(points, gradients, self.padding, orientation)

        # Sections of the path provided. Spans are authored against the points
        # as given, so they shift along with them.
        self.sections = SectionSchedule(
            (
                SectionSpan(
                    span.path_start + self.padding,
                    span.path_stop + self.padding,
                    span.section,
                )
                for span in spans
            ),
            len(self.path.path_points),
        )

        # How far the sections extend either side of the path, at any turn
        self.reach = lateral_reach(sections)

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
