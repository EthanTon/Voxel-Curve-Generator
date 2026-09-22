import unittest
from math import cos, radians, sin

from src.Core.CrossSection.annotations import FeatureAnnotation
from src.Core.CrossSection.cross_section import CrossSection, PointField
from src.Core.CrossSection.schedule import SectionSpan
from src.Core.Domain import Domain
from src.Core.Feature.feature import PointFeature
from src.Core.Field.block_field import BlockField
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.modes import PlacementMode, StepMode
from src.Generation.Geometry.boundaries import LateralReach
from src.Generation.Geometry.path_handler import PathGeometry
from src.Generation.Geometry.reach import feature_lateral_span, lateral_reach


def feature(block_xs=(), offset=None):
    # A feature with one stone block at each local x in `block_xs`.
    registry = BlockRegistry()
    stone = registry.register(BlockState.create("minecraft:stone"))
    blocks = BlockField(registry.palette)
    for x in block_xs:
        blocks.set_block_id((x, 0, 0), stone)
    return PointFeature(blocks, PlacementMode.OVERWRITE, Domain(1, "test"), offset)


def section(*placements, origin=(0, 0, 0)):
    # A section with an annotation at each (local x, feature) placement.
    return CrossSection(
        PointField(
            FeatureAnnotation(f"a{i}", (x, 0, 0), built, StepMode.CHEBYSHEV)
            for i, (x, built) in enumerate(placements)
        ),
        origin=origin,
    )


class FeatureLateralSpanTests(unittest.TestCase):
    def test_an_empty_feature_covers_only_its_position(self) -> None:
        self.assertEqual(feature_lateral_span(feature()), (0, 0))

    def test_blocks_widen_the_span(self) -> None:
        self.assertEqual(feature_lateral_span(feature((-1, 0, 2))), (-1, 2))

    def test_the_offset_shifts_the_span(self) -> None:
        self.assertEqual(feature_lateral_span(feature((-1, 1), offset=(3, 0, 0))), (2, 4))


class LateralReachTests(unittest.TestCase):
    def test_no_sections_reach_nowhere(self) -> None:
        self.assertEqual(lateral_reach([]), LateralReach(0.0, 0.0))

    def test_each_side_is_measured_separately(self) -> None:
        wide_right = section((-2, feature()), (5, feature()))
        self.assertEqual(lateral_reach([wide_right], 90), LateralReach(2.0, 5.0))

    def test_every_quarter_turn_reaches_the_same(self) -> None:
        wide_right = section((-2, feature()), (5, feature()))
        for turn in (0, 90, 180, 270):
            self.assertEqual(lateral_reach([wide_right], turn), LateralReach(2.0, 5.0))

    def test_a_section_on_one_side_does_not_reach_the_other(self) -> None:
        self.assertEqual(lateral_reach([section((3, feature()))], 90), LateralReach(0.0, 3.0))

    def test_positions_are_measured_from_the_origin(self) -> None:
        centred = section((2, feature()), origin=(2, 0, 0))
        self.assertEqual(lateral_reach([centred], 90), LateralReach(0.0, 0.0))

    def test_feature_blocks_and_offsets_extend_the_reach(self) -> None:
        self.assertEqual(
            lateral_reach([section((4, feature((-1, 1))))], 90),
            LateralReach(0.0, 5.0),
        )
        self.assertEqual(
            lateral_reach([section((0, feature((0,), offset=(-3, 0, 0))))], 90),
            LateralReach(3.0, 0.0),
        )

    def test_a_diagonal_reaches_less_far_to_the_side_than_its_quarters(self) -> None:
        # A join is the two quarters carried along the diagonal and then
        # centred on the path point. Carrying something along travel does not
        # move it sideways, and centring pulls it back in, so what is left is
        # a quarter's row foreshortened and no longer lopsided.
        wide_right = section((-2, feature()), (5, feature()))
        quarter = lateral_reach([wide_right], 90)
        diagonal = lateral_reach([wide_right], 135)
        self.assertLess(diagonal.half_width, quarter.half_width)
        self.assertAlmostEqual(diagonal.right, 4 / 2 ** 0.5)

    def test_without_a_turn_it_takes_the_worst_of_every_turn(self) -> None:
        wide_right = section((-2, feature()), (5, feature()))
        worst = lateral_reach([wide_right])
        for turn in range(0, 360, 45):
            each = lateral_reach([wide_right], turn)
            self.assertGreaterEqual(worst.left, each.left)
            self.assertGreaterEqual(worst.right, each.right)

    def test_several_sections_take_the_widest_of_each_side(self) -> None:
        left_heavy = section((-6, feature()), (1, feature()))
        right_heavy = section((-1, feature()), (4, feature()))
        self.assertEqual(
            lateral_reach([left_heavy, right_heavy], 90),
            LateralReach(6.0, 4.0),
        )


class PathGeometryReachTests(unittest.TestCase):
    def quarter_circle(self):
        # Radius 10, curving right: orthogonal planes meet near its centre.
        angles = [radians(a) for a in (0.0, 45.0, 90.0)]
        points = [(round(10 * cos(a)), 64 + i, round(10 * sin(a))) for i, a in enumerate(angles)]
        gradients = [(-sin(a), 0.0, cos(a)) for a in angles]
        return points, gradients

    def test_the_reach_comes_from_the_scheduled_sections(self) -> None:
        wide_right = section((-2, feature()), (5, feature()))
        geometry = PathGeometry(
            [(0, 64, z) for z in range(10)],
            [(0.0, 0.0, 1.0)] * 10,
            [SectionSpan(0, 9, wide_right)],
        )
        self.assertEqual(geometry.reach, lateral_reach([wide_right]))

    def test_a_section_reaching_past_the_inside_of_a_bend_is_rejected(self) -> None:
        points, gradients = self.quarter_circle()
        inside = section((12, feature()))
        with self.assertRaises(ValueError):
            PathGeometry(points, gradients, [SectionSpan(0, 2, inside)])

    def test_reaching_toward_the_outside_of_a_bend_is_allowed(self) -> None:
        points, gradients = self.quarter_circle()
        outside = section((-12, feature()))
        PathGeometry(points, gradients, [SectionSpan(0, 2, outside)])


if __name__ == "__main__":
    unittest.main()
