import unittest

from src.Core.coordinates import BlockPos
from src.Core.line import line
from src.Core.modes import LineMode

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
            for mode in LineMode:
                with self.subTest(start=start, end=end, mode=mode):
                    run = line(start, end, mode)

                    self.assertEqual(run[0], BlockPos(*start))
                    self.assertEqual(run[-1], BlockPos(*end))

    def test_a_line_to_nowhere_is_the_block_it_started_on(self) -> None:
        for mode in LineMode:
            self.assertEqual(line((3, 4, 5), (3, 4, 5), mode), (BlockPos(3, 4, 5),))

    def test_every_step_moves_one_block_at_most(self) -> None:
        for start, end in CORNERS:
            for mode in LineMode:
                with self.subTest(start=start, end=end, mode=mode):
                    for step in steps(line(start, end, mode)):
                        self.assertLessEqual(max(abs(axis) for axis in step), 1)

    def test_a_chebyshev_line_is_as_short_as_the_longest_axis(self) -> None:
        for start, end in CORNERS:
            with self.subTest(start=start, end=end):
                run = line(start, end, LineMode.CHEBYSHEV)

                longest = max(abs(end[axis] - start[axis]) for axis in range(3))
                self.assertEqual(len(run), longest + 1)

    def test_a_chebyshev_line_may_step_diagonally(self) -> None:
        self.assertEqual(
            line((0, 0, 0), (4, 0, 4)),
            tuple(BlockPos(n, 0, n) for n in range(5)),
        )

    def test_a_rail_line_shares_a_face_in_plan(self) -> None:
        # One block across per step, and the climb rides that step rather
        # than taking one of its own: one along and one up is a slope, which
        # is the only way a rail climbs. A block in the corner instead is a
        # step no rail can express.
        for start, end in CORNERS:
            with self.subTest(start=start, end=end):
                run = line(start, end, LineMode.RAIL)

                for across, up, along in steps(run):
                    self.assertEqual(abs(across) + abs(along), 1, (across, up, along))
                    self.assertLessEqual(abs(up), 1)

    def test_a_climb_with_nothing_across_it_still_steps(self) -> None:
        # Straight up is the one step with nothing across to ride.
        run = line((0, 0, 0), (0, 3, 0), LineMode.RAIL)

        self.assertEqual([p.y for p in run], [0, 1, 2, 3])

    def test_a_manhattan_line_shares_a_face_in_3d(self) -> None:
        for start, end in CORNERS:
            with self.subTest(start=start, end=end):
                run = line(start, end, LineMode.MANHATTAN)
                for step in steps(run):
                    self.assertEqual(sum(abs(axis) for axis in step), 1)
                self.assertEqual(
                    len(run), 1 + sum(abs(end[i] - start[i]) for i in range(3))
                )

        self.assertEqual(
            line((36, 73, -2), (37, 74, -2), LineMode.MANHATTAN),
            (
                BlockPos(36, 73, -2),
                BlockPos(37, 73, -2),
                BlockPos(37, 74, -2),
            ),
        )

    def test_a_rail_line_is_as_long_as_its_plan(self) -> None:
        # Its length is what it covers across, since the climb rides along.
        for start, end in CORNERS:
            with self.subTest(start=start, end=end):
                run = line(start, end, LineMode.RAIL)

                across = abs(end[0] - start[0]) + abs(end[2] - start[2])
                self.assertEqual(len(run), across + 1)

    def test_a_line_never_doubles_back_on_a_block(self) -> None:
        for start, end in CORNERS:
            for mode in LineMode:
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
            for mode in LineMode:
                with self.subTest(start=start, end=end, mode=mode):
                    there = line(start, end, mode)
                    back = line(end, start, mode)

                    self.assertEqual(there, back[::-1])

    def test_manhattan_is_the_stepped_bresenham(self) -> None:
        # The reference's connected line: on a diagonal step it emits the
        # block one along the first axis before the one that moved both.
        self.assertEqual(
            [(p.x, p.z) for p in line((0, 0, 0), (4, 0, 4), LineMode.MANHATTAN)],
            [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4)],
        )
        self.assertEqual(
            [(p.x, p.z) for p in line((0, 0, 0), (4, 0, 2), LineMode.MANHATTAN)],
            [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (4, 1), (4, 2)],
        )

    def test_the_x_step_of_a_diagonal_comes_first(self) -> None:
        # Which is what makes it the reference's line rather than one that
        # picks the axis per step by whichever has travelled furthest.
        run = [(p.x, p.z) for p in line((0, 0, 0), (5, 0, 5), LineMode.MANHATTAN)]
        for (ax, az), (bx, bz) in zip(run, run[1:]):
            if az != bz:
                self.assertNotEqual((ax, az), (0, 0), run)

    def test_an_unknown_line_mode_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown line mode"):
            line((0, 0, 0), (1, 1, 1), "sideways")


if __name__ == "__main__":
    unittest.main()
