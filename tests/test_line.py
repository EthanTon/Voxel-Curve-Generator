import unittest

from src.Core.coordinates import BlockPos
from src.Core.line import line
from src.Core.modes import StepMode

CORNERS = (
    ((0, 0, 0), (5, 0, 0)),
    ((0, 0, 0), (5, 0, 5)),
    ((0, 0, 0), (10, 3, 7)),
    ((0, 0, 0), (-10, -3, 7)),
    ((4, 70, -2), (-3, 64, 9)),
    ((-8, 60, 12), (-8, 60, -12)),
)


def steps(run):
    return [
        tuple(after[axis] - before[axis] for axis in range(3))
        for before, after in zip(run, run[1:])
    ]


class LineTests(unittest.TestCase):
    def test_a_line_runs_from_one_end_to_the_other(self) -> None:
        for start, end in CORNERS:
            for mode in StepMode:
                with self.subTest(start=start, end=end, mode=mode):
                    run = line(start, end, mode)

                    self.assertEqual(run[0], BlockPos(*start))
                    self.assertEqual(run[-1], BlockPos(*end))

    def test_a_line_to_nowhere_is_the_block_it_started_on(self) -> None:
        for mode in StepMode:
            self.assertEqual(line((3, 4, 5), (3, 4, 5), mode), (BlockPos(3, 4, 5),))

    def test_every_step_moves_one_block_at_most(self) -> None:
        for start, end in CORNERS:
            for mode in StepMode:
                with self.subTest(start=start, end=end, mode=mode):
                    for step in steps(line(start, end, mode)):
                        self.assertLessEqual(max(abs(axis) for axis in step), 1)

    def test_a_chebyshev_line_is_as_short_as_the_longest_axis(self) -> None:
        for start, end in CORNERS:
            with self.subTest(start=start, end=end):
                run = line(start, end, StepMode.CHEBYSHEV)

                longest = max(abs(end[axis] - start[axis]) for axis in range(3))
                self.assertEqual(len(run), longest + 1)

    def test_a_chebyshev_line_may_step_diagonally(self) -> None:
        self.assertEqual(
            line((0, 0, 0), (4, 0, 4)),
            tuple(BlockPos(n, 0, n) for n in range(5)),
        )

    def test_a_manhattan_line_shares_a_face_between_every_block(self) -> None:
        for start, end in CORNERS:
            with self.subTest(start=start, end=end):
                run = line(start, end, StepMode.MANHATTAN)

                for step in steps(run):
                    self.assertEqual(sum(abs(axis) for axis in step), 1)

    def test_a_manhattan_line_is_as_long_as_every_axis_together(self) -> None:
        for start, end in CORNERS:
            with self.subTest(start=start, end=end):
                run = line(start, end, StepMode.MANHATTAN)

                total = sum(abs(end[axis] - start[axis]) for axis in range(3))
                self.assertEqual(len(run), total + 1)

    def test_a_line_never_doubles_back_on_a_block(self) -> None:
        for start, end in CORNERS:
            for mode in StepMode:
                with self.subTest(start=start, end=end, mode=mode):
                    run = line(start, end, mode)

                    self.assertEqual(len(set(run)), len(run))

    def test_a_shallow_climb_is_spread_along_the_run(self) -> None:
        # Two blocks of rise over six of run, a step every third block.
        self.assertEqual(
            [position.y for position in line((0, 0, 0), (6, 2, 0))],
            [0, 0, 1, 1, 1, 2, 2],
        )

    def test_a_line_drawn_backwards_is_the_same_line(self) -> None:
        # Bresenham breaks its ties on the side it started from, so the ends
        # are ordered before the run is drawn.
        for start, end in CORNERS:
            for mode in StepMode:
                with self.subTest(start=start, end=end, mode=mode):
                    there = line(start, end, mode)
                    back = line(end, start, mode)

                    self.assertEqual(there, back[::-1])

    def test_an_unknown_step_mode_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown step mode"):
            line((0, 0, 0), (1, 1, 1), "sideways")


if __name__ == "__main__":
    unittest.main()
