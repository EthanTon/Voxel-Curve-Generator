import unittest

from src.Core.CrossSection.annotations import FeatureAnnotation
from src.Core.CrossSection.cross_section import CrossSection, PointField
from src.Core.CrossSection.schedule import SectionSpan
from src.Core.Domain import Domain
from src.Core.Feature.feature import PointFeature
from src.Core.Field.block_field import BlockField
from src.Core.Path.path import Path
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.modes import PlacementMode, StepMode
from src.Generation.Geometry.padding import (
    confirm_orthogonal_ends,
    extend_ends,
    padded_path,
)
from src.Generation.Geometry.path_handler import PathGeometry
from src.Generation.Geometry.reach import (
    lateral_reach,
    feature_longitudinal_span,
    longitudinal_reach,
    required_lead,
)


def feature(blocks_at=(), offset=None):
    # A feature with one stone block at each local position in `blocks_at`.
    registry = BlockRegistry()
    stone = registry.register(BlockState.create("minecraft:stone"))
    blocks = BlockField(registry.palette)
    for position in blocks_at:
        blocks.set_block_id(position, stone)
    return PointFeature(blocks, PlacementMode.OVERWRITE, Domain(1, "test"), offset)


def section(*placements, origin=(0, 0, 0)):
    # A section with an annotation at each (local position, feature) placement.
    return CrossSection(
        PointField(
            FeatureAnnotation(f"a{i}", position, built, StepMode.CHEBYSHEV)
            for i, (position, built) in enumerate(placements)
        ),
        origin=origin,
    )


def straight(count, spacing=1):
    # Points along +z, `spacing` blocks apart, all heading +z.
    return (
        [(0, 64, spacing * z) for z in range(count)],
        [(0.0, 0.0, 1.0)] * count,
    )


class FeatureLongitudinalSpanTests(unittest.TestCase):
    def test_an_empty_feature_covers_only_its_position(self) -> None:
        self.assertEqual(feature_longitudinal_span(feature()), (0, 0))

    def test_blocks_deepen_the_span(self) -> None:
        self.assertEqual(
            feature_longitudinal_span(feature([(0, 0, -1), (0, 0, 2)])), (-1, 2)
        )

    def test_the_offset_shifts_the_span(self) -> None:
        self.assertEqual(
            feature_longitudinal_span(
                feature([(0, 0, -1), (0, 0, 1)], offset=(0, 0, 3))
            ),
            (2, 4),
        )


class LongitudinalReachTests(unittest.TestCase):
    def test_no_sections_reach_nowhere(self) -> None:
        self.assertEqual(longitudinal_reach([]), 0.0)

    def test_a_position_ahead_reaches_that_far(self) -> None:
        deep = section(((0, 0, 3), feature()))
        self.assertEqual(longitudinal_reach([deep], 0), 3.0)
        # A join is centred on the path point, and a section one column wide
        # has one copy of each arm, so the two land on it and reach nothing.
        self.assertAlmostEqual(longitudinal_reach([deep], 45), 0.0)

    def test_blocks_reach_where_no_position_does(self) -> None:
        # This section's one position sits on the path point, and its blocks
        # are what reach forward. At a quarter the positions alone are all
        # CrossSection.reach sees, so it sees nothing; longitudinal_reach
        # widens them by the blocks and sees two.
        blocky = section(((0, 0, 0), feature([(0, 0, -2), (0, 0, 2)])))
        self.assertEqual(blocky.reach(0), 0.0)
        self.assertEqual(blocky.reach(90), 0.0)
        self.assertEqual(longitudinal_reach([blocky], 0), 2.0)

    def test_a_section_is_as_long_as_the_blocks_it_paints(self) -> None:
        # A whole profile carried by one annotation is still a profile, so
        # the section measures the blocks rather than the bare position, and
        # a diagonal has half of that to stagger by.
        blocky = section(((0, 0, 0), feature([(0, 0, -2), (0, 0, 2)])))

        self.assertEqual(blocky.long_length, 5)


class RequiredLeadTests(unittest.TestCase):
    def test_a_path_with_no_sections_needs_no_lead(self) -> None:
        self.assertEqual(required_lead(Path(*straight(6)), []), 0)

    def test_depth_along_the_path_sets_the_lead(self) -> None:
        # Three deep, and no layout puts it further along travel than that,
        # so the lead has to cover three.
        deep = section(((0, 0, 3), feature()))
        self.assertEqual(required_lead(Path(*straight(6)), [deep]), 3)

    def test_reach_to_the_side_sets_the_lead_when_it_is_greater(self) -> None:
        # Boundaries converge on a bend, so a section reaching further to the
        # side than ahead still spills past them. This one is a single column
        # four out and one deep: a diagonal turns some of that four along
        # travel, but never as much as the four it reaches sideways.
        wide = section(((4, 0, 0), feature()))
        self.assertEqual(longitudinal_reach([wide]), 0.0)
        self.assertEqual(lateral_reach([wide]).half_width, 4.0)
        self.assertEqual(required_lead(Path(*straight(6)), [wide]), 4)

    def test_the_lead_is_counted_in_points_not_blocks(self) -> None:
        # Points two blocks apart cover the same reach in half the points.
        deep = section(((0, 0, 3), feature()))
        self.assertEqual(required_lead(Path(*straight(6)), [deep]), 3)
        self.assertEqual(required_lead(Path(*straight(6, spacing=2)), [deep]), 2)


class ExtendEndsTests(unittest.TestCase):
    def test_no_lead_changes_nothing(self) -> None:
        positions, gradients = extend_ends(*straight(4), 0)
        self.assertEqual([tuple(p) for p in positions], [(0, 64, z) for z in range(4)])
        self.assertEqual(len(gradients), 4)

    def test_a_lead_extends_both_ends_by_the_last_interval(self) -> None:
        positions, _ = extend_ends(*straight(4), 2)
        self.assertEqual(
            [tuple(p) for p in positions],
            [(0, 64, z) for z in range(-2, 6)],
        )

    def test_a_diagonal_lead_lands_on_whole_blocks(self) -> None:
        points = [(x, 64, x) for x in range(3)]
        positions, _ = extend_ends(points, [(1.0, 0.0, 1.0)] * 3, 2)
        self.assertEqual(
            [tuple(p) for p in positions],
            [(x, 64, x) for x in range(-2, 5)],
        )

    def test_a_lead_continues_a_climb(self) -> None:
        points = [(0, 64, 0), (0, 65, 1), (0, 66, 2)]
        positions, _ = extend_ends(points, [(0.0, 1.0, 1.0)] * 3, 1)
        self.assertEqual(
            [tuple(p) for p in positions],
            [(0, 63, -1), (0, 64, 0), (0, 65, 1), (0, 66, 2), (0, 67, 3)],
        )

    def test_padding_carries_the_gradient_of_the_end_it_extends(self) -> None:
        points = [(0, 64, 0), (0, 64, 1), (1, 64, 2)]
        gradients = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]
        _, extended = extend_ends(points, gradients, 1)
        self.assertEqual(tuple(extended[0]), (0.0, 0.0, 1.0))
        self.assertEqual(tuple(extended[-1]), (1.0, 0.0, 1.0))

    def test_a_single_point_cannot_say_which_way_its_ends_run(self) -> None:
        with self.assertRaises(ValueError):
            extend_ends([(0, 64, 0)], [(0.0, 0.0, 1.0)], 1)

    def test_a_repeated_end_point_leaves_no_interval(self) -> None:
        with self.assertRaises(ValueError):
            extend_ends([(0, 64, 0), (0, 64, 0)], [(0.0, 0.0, 1.0)] * 2, 1)

    def test_a_negative_lead_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extend_ends(*straight(4), -1)


class ConfirmOrthogonalEndsTests(unittest.TestCase):
    def test_a_horizontal_end_is_confirmed(self) -> None:
        self.assertIsNone(confirm_orthogonal_ends(*straight(4)))

    def test_a_vertical_first_point_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "first point.*lead-in"):
            confirm_orthogonal_ends(
                [(0, 64, 0), (0, 64, 1)], [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            )

    def test_a_vertical_last_point_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "last point.*lead-out"):
            confirm_orthogonal_ends(
                [(0, 64, 0), (0, 64, 1)], [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0)]
            )

    def test_an_empty_path_has_no_ends(self) -> None:
        with self.assertRaises(ValueError):
            confirm_orthogonal_ends([], [])


class PaddedPathTests(unittest.TestCase):
    def test_the_authored_points_become_the_path_proper(self) -> None:
        path = padded_path(*straight(4), 2)
        self.assertEqual(len(path.path_points), 8)
        self.assertEqual((path.first_index, path.last_index), (2, 5))
        self.assertEqual(
            [p.position.z for p in path.bounded_points], [0, 1, 2, 3]
        )
        self.assertEqual(path.length(), 3.0)

    def test_a_vertical_end_is_refused_before_anything_is_extended(self) -> None:
        with self.assertRaises(ValueError):
            padded_path([(0, 64, 0), (0, 65, 0)], [(0.0, 1.0, 0.0)] * 2, 1)


class PathGeometryPaddingTests(unittest.TestCase):
    def test_a_path_with_no_sections_is_not_padded(self) -> None:
        geometry = PathGeometry(*straight(4))
        self.assertEqual(geometry.padding, 0)
        self.assertEqual((geometry.path.first_index, geometry.path.last_index), (0, 3))

    def test_the_padding_is_computed_from_what_the_sections_reach(self) -> None:
        deep = section(((0, 0, 3), feature()))
        geometry = PathGeometry(
            *straight(6), section_spans=[SectionSpan(0, 5, deep)]
        )
        self.assertEqual(geometry.padding, 3)
        self.assertEqual(len(geometry.path.path_points), 12)
        self.assertEqual((geometry.path.first_index, geometry.path.last_index), (3, 8))

    def test_the_authored_points_are_all_on_the_path_proper(self) -> None:
        deep = section(((0, 0, 3), feature()))
        geometry = PathGeometry(
            *straight(6), section_spans=[SectionSpan(0, 5, deep)]
        )
        self.assertEqual(
            [p.position.z for p in geometry.path.bounded_points], list(range(6))
        )

    def test_section_spans_shift_with_the_padding(self) -> None:
        # Spans are authored against the points as given, so a span over every
        # authored point still covers every point of the path proper.
        deep = section(((0, 0, 3), feature()))
        geometry = PathGeometry(
            *straight(6), section_spans=[SectionSpan(1, 4, deep)]
        )
        [span] = geometry.sections.spans
        self.assertEqual((span.path_start, span.path_stop), (4, 7))
        self.assertIs(geometry.sections.at(4), deep)
        self.assertIsNone(geometry.sections.at(3))

    def test_padding_can_be_overridden(self) -> None:
        deep = section(((0, 0, 3), feature()))
        geometry = PathGeometry(
            *straight(6), section_spans=[SectionSpan(0, 5, deep)], padding=1
        )
        self.assertEqual(geometry.padding, 1)
        self.assertEqual((geometry.path.first_index, geometry.path.last_index), (1, 6))


if __name__ == "__main__":
    unittest.main()
