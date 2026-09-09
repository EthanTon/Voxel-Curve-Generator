import unittest

from src.Core.Path.path import PathPoint
from src.Core.Path.path_frame import OrthogonalBoundary, PathFrame
from src.Core.coordinates import BlockPos, Vec2, Vec3


def boundary(index=0, origin=(0.0, 0.0), tangent=(0.0, 1.0)):
    return OrthogonalBoundary.from_tangent(index, origin, tangent)


class OrthogonalBoundaryTests(unittest.TestCase):
    def test_side_is_signed_along_the_normal(self) -> None:
        b = boundary(origin=(0.0, 10.0), tangent=(0.0, 1.0))
        self.assertGreater(b.side(0, 11), 0.0)
        self.assertLess(b.side(0, 9), 0.0)
        self.assertEqual(b.side(0, 10), 0.0)

    def test_the_plane_itself_belongs_to_the_frame_after_it(self) -> None:
        b = boundary(origin=(0.0, 10.0), tangent=(0.0, 1.0))
        self.assertTrue(b.after(0, 10))

    def test_height_never_affects_membership(self) -> None:
        b = boundary(origin=(5.0, 5.0), tangent=(1.0, 1.0))
        # side takes only (x, z); the same column answers the same at any height
        self.assertEqual(b.side(7, 3), b.side(7, 3))

    def test_from_tangent_normalises(self) -> None:
        b = boundary(tangent=(0.0, 7.0))
        self.assertAlmostEqual(b.normal.z, 1.0)

    def test_zero_normal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OrthogonalBoundary(0, Vec2(0.0, 0.0), Vec2(0.0, 0.0))
        with self.assertRaises(ValueError):
            OrthogonalBoundary.from_tangent(0, (0.0, 0.0), (0.0, 0.0))

    def test_parallel_detection(self) -> None:
        a = boundary(tangent=(1.0, 0.0))
        b = boundary(origin=(3.0, 0.0), tangent=(-1.0, 0.0))
        c = boundary(tangent=(0.0, 1.0))
        self.assertTrue(a.is_parallel_to(b))
        self.assertFalse(a.is_parallel_to(c))


class GradientSourcedBoundaryTests(unittest.TestCase):
    def test_normal_comes_from_the_gradient_horizontal_projection(self) -> None:
        point = PathPoint(BlockPos(10, 64, 20), Vec3(0.0, 0.5, 3.0))
        b = OrthogonalBoundary.from_path_point(7, point)
        self.assertEqual(b.path_index, 7)
        self.assertEqual(b.origin, Vec2(10, 20))
        self.assertAlmostEqual(b.normal.x, 0.0)
        self.assertAlmostEqual(b.normal.z, 1.0)

    def test_vertical_component_does_not_tilt_the_plane(self) -> None:
        flat = PathPoint(BlockPos(0, 64, 0), Vec3(1.0, 0.0, 0.0))
        steep = PathPoint(BlockPos(0, 64, 0), Vec3(1.0, 9.0, 0.0))
        a = OrthogonalBoundary.from_path_point(0, flat)
        b = OrthogonalBoundary.from_path_point(0, steep)
        self.assertAlmostEqual(a.normal.x, b.normal.x)
        self.assertAlmostEqual(a.normal.z, b.normal.z)

    def test_purely_vertical_gradient_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            OrthogonalBoundary.from_path_point(0, PathPoint(BlockPos(0, 0, 0), Vec3(0.0, 1.0, 0.0)))

    def test_grade_is_rise_over_run(self) -> None:
        self.assertAlmostEqual(PathPoint(BlockPos(0, 0, 0), Vec3(3.0, 3.0, 4.0)).grade, 0.6)
        self.assertAlmostEqual(PathPoint(BlockPos(0, 0, 0), Vec3(1.0, 0.0, 0.0)).grade, 0.0)

    def test_grade_undefined_when_vertical(self) -> None:
        point = PathPoint(BlockPos(0, 0, 0), Vec3(0.0, 2.0, 0.0))
        self.assertTrue(point.is_vertical)
        with self.assertRaises(ValueError):
            point.grade


class PathFrameTests(unittest.TestCase):
    def test_adjacent_frames_tile_without_overlap_or_gap(self) -> None:
        shared = boundary(index=10, origin=(0.0, 10.0), tangent=(0.0, 1.0))
        lower = PathFrame(0, 0, 10, None, shared, y_offset=64)
        upper = PathFrame(1, 10, 20, shared, None, y_offset=65)

        for x in range(-8, 9):
            for z in range(-8, 25):
                owners = [f for f in (lower, upper) if f.contains(x, z)]
                self.assertEqual(len(owners), 1, f"column ({x}, {z}) owned by {len(owners)}")

    def test_a_diagonal_boundary_splits_a_column_run(self) -> None:
        shared = boundary(index=5, origin=(0.0, 0.0), tangent=(1.0, 1.0))
        before = PathFrame(0, 0, 5, None, shared, y_offset=64)
        after = PathFrame(1, 5, 10, shared, None, y_offset=65)
        row = [after.contains(x, 0) for x in range(-3, 4)]
        self.assertEqual(row, [False, False, False, True, True, True, True])
        self.assertTrue(all(before.contains(x, 0) != after.contains(x, 0) for x in range(-3, 4)))

    def test_open_ended_frames(self) -> None:
        frame = PathFrame(0, 0, 5, None, None, y_offset=64)
        self.assertTrue(frame.is_open_ended)
        self.assertTrue(frame.contains(10_000, -10_000))

    def test_reversed_path_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PathFrame(0, 9, 4, None, None, y_offset=64)

    def test_degenerate_frame_is_detected(self) -> None:
        start = boundary(index=3, origin=(0.0, 10.0), tangent=(0.0, 1.0))
        # end plane sits behind the start plane: no column can satisfy both
        end = boundary(index=3, origin=(0.0, 4.0), tangent=(0.0, 1.0))
        frame = PathFrame(0, 3, 3, start, end, y_offset=64)
        self.assertTrue(frame.is_degenerate)
        self.assertFalse(any(frame.contains(0, z) for z in range(-20, 40)))

    def test_healthy_frame_is_not_degenerate(self) -> None:
        start = boundary(index=0, origin=(0.0, 0.0), tangent=(0.0, 1.0))
        end = boundary(index=8, origin=(0.0, 8.0), tangent=(0.0, 1.0))
        self.assertFalse(PathFrame(0, 0, 8, start, end, y_offset=64).is_degenerate)


if __name__ == "__main__":
    unittest.main()
