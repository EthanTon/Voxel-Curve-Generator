import unittest

from src.Core.Path.boundary import Boundary
from src.Core.Path.path_frame import PathFrame, frames_for_index


def boundary(index=0, origin=(0.0, 0.0), tangent=(0.0, 1.0)):
    return Boundary.from_direction(origin, tangent, path_index=index)


class PathFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        # A straight path along +z from z=0 to z=20, stepping up at z=10.
        self.path_start = boundary(index=0, origin=(0.0, 0.0))
        self.shared = boundary(index=10, origin=(0.0, 10.0))
        self.path_end = boundary(index=20, origin=(0.0, 20.0))
        self.lower = PathFrame(0, 0, 10, self.path_start, self.shared, y_offset=64)
        self.upper = PathFrame(1, 10, 20, self.shared, self.path_end, y_offset=65)

    def owners(self, x: int, z: int) -> list[PathFrame]:
        return [f for f in (self.lower, self.upper) if f.contains(x, z)]

    def test_adjacent_frames_tile_without_overlap_or_gap(self) -> None:
        for x in range(-8, 9):
            for z in range(0, 20):
                self.assertEqual(len(self.owners(x, z)), 1, f"column ({x}, {z})")

    def test_path_ends_bound_like_any_other_boundary(self) -> None:
        # The start plane belongs to the first frame, exactly as the shared
        # plane belongs to the frame after it.
        self.assertEqual(self.owners(0, 0), [self.lower])
        self.assertEqual(self.owners(0, 10), [self.upper])
        # The end plane belongs to the side after it, which is off the path.
        self.assertEqual(self.owners(0, 19), [self.upper])
        self.assertEqual(self.owners(0, 20), [])

    def test_nothing_outside_the_path_ends_is_owned(self) -> None:
        for z in (-10_000, -1, 21, 10_000):
            self.assertEqual(self.owners(0, z), [])

    def test_a_diagonal_boundary_splits_a_column_run(self) -> None:
        start = boundary(index=0, origin=(-10.0, -10.0), tangent=(1.0, 1.0))
        shared = boundary(index=5, origin=(0.0, 0.0), tangent=(1.0, 1.0))
        end = boundary(index=10, origin=(10.0, 10.0), tangent=(1.0, 1.0))
        before = PathFrame(0, 0, 5, start, shared, y_offset=64)
        after = PathFrame(1, 5, 10, shared, end, y_offset=65)
        row = [after.contains(x, 0) for x in range(-3, 4)]
        self.assertEqual(row, [False, False, False, True, True, True, True])
        self.assertTrue(all(before.contains(x, 0) != after.contains(x, 0) for x in range(-3, 4)))

    def test_reversed_path_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PathFrame(0, 9, 4, self.path_start, self.path_end, y_offset=64)

    def test_degenerate_frame_is_detected(self) -> None:
        start = boundary(index=3, origin=(0.0, 10.0))
        # end plane sits behind the start plane: no column can satisfy both
        end = boundary(index=3, origin=(0.0, 4.0))
        frame = PathFrame(0, 3, 3, start, end, y_offset=64)
        self.assertTrue(frame.is_degenerate)
        self.assertFalse(any(frame.contains(0, z) for z in range(-20, 40)))

    def test_healthy_frame_is_not_degenerate(self) -> None:
        self.assertFalse(self.lower.is_degenerate)
        self.assertFalse(self.upper.is_degenerate)



class FramesForIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        # Path proper 2-21, in frames 2-8, 8-15 and 15-21.
        self.frames = [
            PathFrame(0, 2, 8, boundary(2, (0.0, 2.0)), boundary(8, (0.0, 8.0)), 64),
            PathFrame(1, 8, 15, boundary(8, (0.0, 8.0)), boundary(15, (0.0, 15.0)), 65),
            PathFrame(2, 15, 21, boundary(15, (0.0, 15.0)), boundary(21, (0.0, 21.0)), 66),
        ]

    def positions(self, index: int, lead_in: int = 0, lead_out: int = 0) -> list[int]:
        return [f.index for f in frames_for_index(self.frames, index, lead_in, lead_out)]

    def test_without_a_lead_a_point_on_a_break_goes_to_both_frames(self) -> None:
        self.assertEqual(self.positions(7), [0])
        self.assertEqual(self.positions(8), [0, 1])
        self.assertEqual(self.positions(9), [1])
        self.assertEqual(self.positions(15), [1, 2])

    def test_a_lead_reaches_into_the_neighbouring_frames(self) -> None:
        # Frame 1 (8-15) with 2 in and 1 out takes points 6-16.
        self.assertEqual(self.positions(6, lead_in=2, lead_out=1), [0, 1])
        self.assertEqual(self.positions(5, lead_in=2, lead_out=1), [0])
        self.assertEqual(self.positions(16, lead_in=2, lead_out=1), [1, 2])
        self.assertEqual(self.positions(17, lead_in=2, lead_out=1), [2])

    def test_padding_is_pasted_only_as_far_as_the_lead(self) -> None:
        self.assertEqual(self.positions(1), [])
        self.assertEqual(self.positions(1, lead_in=1), [0])
        self.assertEqual(self.positions(0, lead_in=1), [])
        self.assertEqual(self.positions(24, lead_out=3), [2])

    def test_bad_input(self) -> None:
        with self.assertRaises(ValueError):
            frames_for_index([], 0)
        with self.assertRaises(ValueError):
            frames_for_index(self.frames, 0, lead_in=-1)


if __name__ == "__main__":
    unittest.main()
