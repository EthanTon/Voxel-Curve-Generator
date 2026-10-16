import unittest
from math import cos, radians, sin

from src.Core.Path.boundary import Boundary
from src.Core.Path.path import Path, PathPoint
from src.Core.Path.path_frame import PathFrame
from src.Core.coordinates import BlockPos, Vec2, Vec3
from src.Generation.Geometry.boundaries import (
    boundary_at,
    crosses_within,
    crossing_offsets,
    find_crossings,
    orthogonal,
    path_bounds,
    validate_frames,
)


def straight(count: int, **padding) -> Path:
    return Path([(0, 64, z) for z in range(count)], [(0.0, 0.0, 1.0)] * count, **padding)


def plane(origin=(0.0, 0.0), direction=(0.0, 1.0), index=None) -> Boundary:
    return Boundary.from_direction(origin, direction, path_index=index)


class OrientationTests(unittest.TestCase):
    def test_orthogonal_faces_the_gradient_heading(self) -> None:
        path = Path([(10, 64, 20)], [(0.0, 0.5, 3.0)])
        b = boundary_at(path, 0)
        self.assertEqual(b.path_index, 0)
        self.assertEqual(b.origin, Vec2(10, 20))
        self.assertAlmostEqual(b.normal.x, 0.0)
        self.assertAlmostEqual(b.normal.z, 1.0)

    def test_grade_does_not_tilt_the_plane(self) -> None:
        path = Path([(0, 64, 0), (0, 64, 0)], [(1.0, 0.0, 0.0), (1.0, 9.0, 0.0)])
        self.assertEqual(boundary_at(path, 0).normal, boundary_at(path, 1).normal)

    def test_vertical_gradient_cannot_be_oriented(self) -> None:
        with self.assertRaises(ValueError):
            orthogonal(Path([(0, 0, 0)], [(0.0, 1.0, 0.0)]), 0)

    def test_a_custom_orientation_is_used_as_given(self) -> None:
        east = lambda path, index: Vec2(1.0, 0.0)
        b = boundary_at(straight(3), 1, orientation=east)
        self.assertEqual(b.normal, Vec2(1.0, 0.0))
        self.assertEqual(b.origin, Vec2(0, 1))

    def test_index_out_of_range(self) -> None:
        with self.assertRaises(IndexError):
            boundary_at(straight(3), 3)


class PathBoundsBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        # z 0-1 lead in, z 9-11 lead out, path proper z 2-8.
        self.path = straight(12, lead_in=2, lead_out=3)
        self.bounds = path_bounds(self.path)

    def test_the_start_plane_sits_half_a_block_before_the_first_point(self) -> None:
        # Every column the plane through the first point passes through is
        # kept, so a slanted start is not notched where it just misses one.
        self.assertEqual(self.bounds.start.path_index, 2)
        self.assertEqual(self.bounds.start.origin, Vec2(0, 1.5))
        self.assertTrue(self.bounds.contains(0, 2))
        self.assertFalse(self.bounds.contains(0, 1))

    def test_the_end_plane_sits_a_block_past_the_last(self) -> None:
        # A frame holds the columns at or after its start and before its end,
        # so an end plane on the last point would leave that point outside.
        # It is not a plane at a path point, and carries no index.
        self.assertEqual(self.bounds.end.origin, Vec2(0, 9))
        self.assertIsNone(self.bounds.end.path_index)

    def test_both_path_ends_are_inside(self) -> None:
        for index in (self.path.first_index, self.path.last_index):
            position = self.path.path_points[index].position
            self.assertTrue(self.bounds.contains(position.x, position.z))

    def test_padding_columns_fall_outside(self) -> None:
        for index, point in enumerate(self.path.path_points):
            if self.path.is_padding(index):
                self.assertFalse(self.bounds.contains(point.position.x, point.position.z))

    def test_cut_is_straight_across_the_path_width(self) -> None:
        for x in range(-6, 7):
            self.assertEqual(
                [self.bounds.contains(x, z) for z in range(12)],
                [self.bounds.contains(0, z) for z in range(12)],
            )

    def test_a_single_point_path_owns_its_own_column(self) -> None:
        bounds = path_bounds(Path([(0, 0, 0)], [(1.0, 0.0, 0.0)]))
        self.assertFalse(bounds.is_degenerate)
        self.assertTrue(bounds.contains(0, 0))
        self.assertFalse(bounds.contains(1, 0))

    def test_empty_path_cannot_be_bounded(self) -> None:
        with self.assertRaises(ValueError):
            path_bounds(Path([], []))


class CrossingTests(unittest.TestCase):
    def test_parallel_planes_never_cross(self) -> None:
        self.assertIsNone(crossing_offsets(plane((0.0, 0.0)), plane((0.0, 5.0))))
        self.assertFalse(crosses_within(plane((0.0, 0.0)), plane((0.0, 5.0)), 1_000.0))

    def test_offsets_locate_the_meeting_point(self) -> None:
        a = plane((0.0, 0.0), (0.0, 1.0))   # the line z = 0
        b = plane((10.0, 10.0), (1.0, 0.0))  # the line x = 10
        along_a, along_b = crossing_offsets(a, b)
        # both planes reach (10, 0) ten blocks from their origins
        self.assertAlmostEqual(abs(along_a), 10.0)
        self.assertAlmostEqual(abs(along_b), 10.0)
        self.assertFalse(crosses_within(a, b, 9.5))
        self.assertTrue(crosses_within(a, b, 10.0))

    def bend(self, radius: float, degrees: float) -> tuple[Boundary, Boundary]:
        # Orthogonal planes at two points on a circle of `radius` about the origin.
        planes = []
        for angle in (0.0, radians(degrees)):
            origin = (radius * cos(angle), radius * sin(angle))
            tangent = (-sin(angle), cos(angle))
            planes.append(plane(origin, tangent))
        return planes[0], planes[1]

    def test_orthogonal_planes_on_a_bend_meet_at_its_centre(self) -> None:
        a, b = self.bend(radius=6.0, degrees=10.0)
        along_a, along_b = crossing_offsets(a, b)
        self.assertAlmostEqual(abs(along_a), 6.0)
        self.assertAlmostEqual(abs(along_b), 6.0)

    def test_a_section_wider_than_the_bend_crosses(self) -> None:
        a, b = self.bend(radius=6.0, degrees=10.0)
        self.assertTrue(crosses_within(a, b, half_width=7.0))
        self.assertFalse(crosses_within(a, b, half_width=5.0))

    def test_find_crossings_reports_consecutive_pairs(self) -> None:
        a, b = self.bend(radius=6.0, degrees=10.0)
        c = plane((100.0, 100.0), b.normal)
        self.assertEqual(find_crossings([a, b, c], half_width=7.0), [(0, 1)])


class ValidateFramesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = straight(21)
        self.start = boundary_at(self.path, 0)
        self.shared = boundary_at(self.path, 10)
        self.end = boundary_at(self.path, 20)

    def frames(self, **overrides) -> list[PathFrame]:
        lower = dict(index=0, path_start=0, path_stop=10,
                     start_boundary=self.start, end_boundary=self.shared, y_offset=64)
        upper = dict(index=1, path_start=10, path_stop=20,
                     start_boundary=self.shared, end_boundary=self.end, y_offset=65)
        upper.update(overrides)
        return [PathFrame(**lower), PathFrame(**upper)]

    def test_a_well_formed_sequence_passes(self) -> None:
        validate_frames(self.path, self.frames(), half_width=50.0)

    def test_empty(self) -> None:
        with self.assertRaises(ValueError):
            validate_frames(self.path, [])

    def test_out_of_order_index(self) -> None:
        with self.assertRaises(ValueError):
            validate_frames(self.path, self.frames(index=5))

    def test_range_outside_the_path(self) -> None:
        with self.assertRaises(ValueError):
            validate_frames(self.path, self.frames(path_stop=21))

    def test_boundary_at_the_wrong_point(self) -> None:
        with self.assertRaises(ValueError):
            validate_frames(self.path, self.frames(end_boundary=boundary_at(self.path, 19)))

    def test_gap_between_frames(self) -> None:
        with self.assertRaises(ValueError):
            validate_frames(
                self.path,
                self.frames(path_start=11, start_boundary=boundary_at(self.path, 11)),
            )

    def test_frames_must_share_their_boundary(self) -> None:
        skewed = Boundary.from_direction((0.0, 10.0), (1.0, 1.0), path_index=10)
        with self.assertRaises(ValueError):
            validate_frames(self.path, self.frames(start_boundary=skewed))

    def test_degenerate_frame(self) -> None:
        with self.assertRaises(ValueError):
            validate_frames(self.path, self.frames(end_boundary=self.shared, path_stop=10))

    def test_boundaries_without_a_path_index_skip_that_check(self) -> None:
        loose = Boundary(self.end.origin, self.end.normal)
        validate_frames(self.path, self.frames(end_boundary=loose))

    def test_crossing_planes_are_rejected_only_when_a_width_is_given(self) -> None:
        tilted = Boundary.from_direction((0.0, 20.0), (1.0, 2.0), path_index=20)
        frames = self.frames(end_boundary=tilted)
        validate_frames(self.path, frames)
        with self.assertRaises(ValueError):
            validate_frames(self.path, frames, half_width=50.0)


if __name__ == "__main__":
    unittest.main()
