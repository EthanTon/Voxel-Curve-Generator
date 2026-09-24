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
    # A padded path with its section schedule and frames. Padding defaults to
    # the lead the sections need.
    def __init__(
        self,
        points,
        gradients,
        section_spans: Iterable[SectionSpan] = (),
        *,
        padding: int | None = None,
        orientation: Orientation = orthogonal,
    ):
        spans = tuple(section_spans)
        sections = tuple(span.section for span in spans)
        authored = Path(points=points, gradients=gradients)

        self.padding = (
            required_lead(authored, sections) if padding is None else padding
        )

        self.path = padded_path(points, gradients, self.padding, orientation)

        # Spans are authored against the unpadded points.
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

        self.reach = lateral_reach(sections)

        start_y = self.path.path_points[self.path.first_index].position.y
        self.elevation_frames = build_frames(
            self.path,
            start_y,
            elevation_breaks(self.path),
            orientation=orientation,
            reach=self.reach,
        )

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

        self.elevation = elevation_from_frames(self.elevation_frames)

        self.bounds = path_bounds(self.path, orientation)
