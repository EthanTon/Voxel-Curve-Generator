import unittest

from src.Core.Feature.feature_path import (
    FeaturePath,
    apart,
    either,
    every,
    spacing_rules,
    turning,
)


def run(points, rules=()):
    # A path of (position, gradient) pairs, one delivery each.
    path = FeaturePath(rules)
    for index, (position, gradient) in enumerate(points):
        path.add(position, gradient, 0, index, 0)
    return path


def straight(length, start=0):
    return [((start + n, 0, 0), (1, 0, 0)) for n in range(length)]


def columns(path):
    return [(point.position.x, point.position.z) for point in path]


class ApartTests(unittest.TestCase):
    def test_the_first_point_is_always_kept(self) -> None:
        self.assertEqual(columns(run(straight(1), (apart(5),))), [(0, 0)])

    def test_points_are_kept_once_they_are_far_enough_along(self) -> None:
        self.assertEqual(
            columns(run(straight(12), (apart(5),))),
            [(0, 0), (5, 0), (10, 0)],
        )

    def test_the_gap_is_measured_from_the_last_point_kept(self) -> None:
        # Not from the last point offered, or every point would be kept.
        self.assertEqual(
            columns(run(straight(7), (apart(3),))), [(0, 0), (3, 0), (6, 0)]
        )

    def test_a_climb_does_not_shorten_the_gap(self) -> None:
        climbing = [((n, n, 0), (1, 1, 0)) for n in range(7)]

        self.assertEqual(
            columns(run(climbing, (apart(3),))), [(0, 0), (3, 0), (6, 0)]
        )


class TurningTests(unittest.TestCase):
    def test_a_straight_run_keeps_only_its_first_point(self) -> None:
        self.assertEqual(columns(run(straight(12), (turning(45),))), [(0, 0)])

    def test_a_bend_earns_a_point(self) -> None:
        bending = [*straight(4), *[((4 + n, 0, -n), (1, 0, -1)) for n in range(1, 5)]]

        self.assertEqual(columns(run(bending, (turning(45),))), [(0, 0), (5, -1)])

    def test_a_shallow_bend_earns_nothing_on_its_own(self) -> None:
        shallow = [*straight(4), *[((4 + n, 0, 0), (10, 0, -1)) for n in range(1, 5)]]

        self.assertEqual(columns(run(shallow, (turning(45),))), [(0, 0)])

    def test_it_holds_a_wire_over_the_track_it_spans(self) -> None:
        # On an arc, a pole every time the track has swung the threshold, so
        # the straight span between poles never strays far from it.
        from math import cos, radians, sin

        arc = [
            (
                (
                    int(round(60 * sin(radians(n)))),
                    0,
                    int(round(-60 * (1 - cos(radians(n))))),
                ),
                (cos(radians(n)), 0.0, -sin(radians(n))),
            )
            for n in range(91)
        ]

        kept = [point.path_index for point in run(arc, (turning(20),))]

        self.assertEqual(kept, [0, 20, 40, 61, 81])

    def test_the_turn_is_measured_from_the_last_point_kept(self) -> None:
        # Two 30 degree steps add up to 60, so the second one earns a point
        # even though neither step is 45 on its own.
        def heading(degrees):
            from math import cos, radians, sin

            return (cos(radians(degrees)), 0.0, -sin(radians(degrees)))

        stepping = [
            ((0, 0, 0), heading(0)),
            ((1, 0, 0), heading(30)),
            ((2, 0, 0), heading(60)),
        ]

        self.assertEqual(columns(run(stepping, (turning(45),))), [(0, 0), (2, 0)])


class EveryTests(unittest.TestCase):
    def test_points_are_counted_rather_than_blocks(self) -> None:
        # Points two blocks apart, kept every three points: six blocks.
        spread = [((n * 2, 0, 0), (1, 0, 0)) for n in range(10)]

        self.assertEqual(
            columns(run(spread, (every(3),))), [(0, 0), (6, 0), (12, 0), (18, 0)]
        )

    def test_the_count_is_taken_from_the_last_point_kept(self) -> None:
        self.assertEqual(
            columns(run(straight(10), (every(4),))), [(0, 0), (4, 0), (8, 0)]
        )


class EitherTests(unittest.TestCase):
    def test_a_point_may_earn_its_place_either_way(self) -> None:
        bending = [*straight(12), *[((12 + n, 0, -n), (1, 0, -1)) for n in range(1, 9)]]

        kept = columns(run(bending, spacing_rules(5, 45)))

        # Spacing puts points along the straight; the bend adds one of its own
        # at 13, and spacing carries on from there.
        self.assertEqual(kept, [(0, 0), (5, 0), (10, 0), (13, -1), (17, -5)])

    def test_rules_are_otherwise_all_required(self) -> None:
        never = lambda path, point: False

        self.assertEqual(len(run(straight(5), (either(never), apart(0)))), 0)

    def test_neither_rule_keeps_every_point(self) -> None:
        self.assertEqual(spacing_rules(0, 0), ())
        self.assertEqual(len(run(straight(5), spacing_rules(0, 0))), 5)

    def test_spacing_alone_ignores_bends(self) -> None:
        self.assertEqual(len(spacing_rules(5, 0)), 1)

    def test_spacing_may_be_counted_in_points(self) -> None:
        spread = [((n * 2, 0, 0), (1, 0, 0)) for n in range(10)]

        by_blocks = columns(run(spread, spacing_rules(6, 0)))
        by_points = columns(run(spread, spacing_rules(6, 0, by_index=True)))

        self.assertEqual(by_blocks, [(0, 0), (6, 0), (12, 0), (18, 0)])
        self.assertEqual(by_points, [(0, 0), (12, 0)])

    def test_an_angle_alone_ignores_distance(self) -> None:
        self.assertEqual(len(spacing_rules(0, 45)), 1)


if __name__ == "__main__":
    unittest.main()
