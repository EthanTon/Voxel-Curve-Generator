import unittest

from src.Core.CrossSection.cross_section import CrossSection
from src.Core.CrossSection.schedule import SectionSpan
from src.Core.Path.elevation import ConstantElevation, MultiElevationStep
from src.Core.Path.path import Path
from src.Generation.Geometry.frames import ElevationBreak, elevation_breaks
from src.Generation.Geometry.path_handler import PathGeometry


def heights(*ys):
    # A straight path heading +z with the given height at each point.
    return [(0, y, z) for z, y in enumerate(ys)], [(0.0, 0.0, 1.0)] * len(ys)


class ElevationBreaksTests(unittest.TestCase):
    def test_a_flat_path_has_no_breaks(self) -> None:
        self.assertEqual(elevation_breaks(Path(*heights(64, 64, 64, 64))), [])

    def test_a_break_at_every_change_in_height(self) -> None:
        path = Path(*heights(64, 64, 65, 65, 63, 63))
        self.assertEqual(
            elevation_breaks(path),
            [ElevationBreak(2, 65), ElevationBreak(4, 63)],
        )

    def test_a_change_at_the_last_point_is_not_a_break(self) -> None:
        self.assertEqual(elevation_breaks(Path(*heights(64, 64, 64, 65))), [])

    def test_padding_is_ignored(self) -> None:
        # Padding heights differ, but only the path proper (64, 64, 65, 65) counts.
        path = Path(*heights(60, 64, 64, 65, 65, 70), lead_in=1, lead_out=1)
        self.assertEqual(elevation_breaks(path), [ElevationBreak(3, 65)])


class PathGeometryTests(unittest.TestCase):
    def test_a_flat_path_is_one_frame_at_its_height(self) -> None:
        geometry = PathGeometry(*heights(70, 70, 70, 70))
        [frame] = geometry.elevation_frames
        self.assertEqual((frame.path_start, frame.path_stop, frame.y_offset), (0, 3, 70))
        self.assertEqual(geometry.elevation, ConstantElevation(70))

    def test_elevation_frames_follow_the_path_height(self) -> None:
        geometry = PathGeometry(*heights(64, 64, 65, 65, 63, 63))
        self.assertEqual(
            [(f.path_start, f.path_stop, f.y_offset) for f in geometry.elevation_frames],
            [(0, 2, 64), (2, 4, 65), (4, 5, 63)],
        )
        self.assertIsInstance(geometry.elevation, MultiElevationStep)
        self.assertEqual(geometry.elevation(0, 1), 64)
        self.assertEqual(geometry.elevation(0, 2), 65)
        self.assertEqual(geometry.elevation(0, 5), 63)

    def test_section_frames_split_at_section_and_elevation_changes(self) -> None:
        road, bridge = CrossSection(), CrossSection()
        geometry = PathGeometry(
            *heights(64, 64, 64, 64, 65, 65, 65, 65, 65, 65),
            section_spans=[SectionSpan(0, 6, road), SectionSpan(7, 9, bridge)],
        )
        self.assertEqual(
            [(f.path_start, f.path_stop, f.y_offset) for f in geometry.section_frames],
            [(0, 4, 64), (4, 7, 65), (7, 9, 65)],
        )

    def test_section_frames_share_the_elevation_boundaries(self) -> None:
        geometry = PathGeometry(*heights(64, 64, 65, 65))
        [lower, upper] = geometry.elevation_frames
        self.assertIs(geometry.section_frames[0].end_boundary, lower.end_boundary)
        self.assertIs(geometry.section_frames[1].start_boundary, upper.start_boundary)

    def test_padding_extends_the_path_past_the_points_given(self) -> None:
        # The points given are the path proper; padding is added around them,
        # so the bounds still fall on the first and last point authored.
        geometry = PathGeometry(*heights(64, 64, 64, 64), padding=2)
        self.assertEqual(len(geometry.path.path_points), 8)
        self.assertEqual((geometry.path.first_index, geometry.path.last_index), (2, 5))
        [frame] = geometry.elevation_frames
        self.assertEqual((frame.path_start, frame.path_stop), (2, 5))
        self.assertEqual(geometry.bounds.start.path_index, 2)
        self.assertEqual(geometry.bounds.end.path_index, 5)
        self.assertTrue(geometry.bounds.contains(0, 2))
        self.assertFalse(geometry.bounds.contains(0, 4))
        self.assertFalse(geometry.bounds.contains(0, -1))


if __name__ == "__main__":
    unittest.main()
