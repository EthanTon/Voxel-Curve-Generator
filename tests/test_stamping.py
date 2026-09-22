import unittest

from src.Core.Domain import DomainRegistry
from src.Core.Feature.feature import PointFeature
from src.Core.Field.block_field import BlockField
from src.Core.Feature.feature_path import FeaturePath
from src.Core.Path.boundary import Boundary
from src.Core.Path.path_frame import PathFrame
from src.Core.World.block_world import BlockWorld, ColumnMask
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, Vec3
from src.Core.modes import ElevationMode, PlacementMode
from src.Stamping import (
    UNCLIPPED,
    laid_offset,
    FixedStamp,
    HeadingStamp,
    Stamp,
    TurnStamp,
    authored_headings,
    authored_turn,
    by_authored,
    heading_of,
    in_frames,
    lateral_turns,
    place,
    snap,
    stamp_along,
    unclipped,
)


class LaidOffsetTests(unittest.TestCase):
    # An offset lands where the section puts its own column at that local x,
    # in the layout that placed the position -- and mirrored the same way
    # that layout is.

    def test_local_x_is_placed_by_the_layout(self) -> None:
        # Two to the right of travel, travelling north, is two east.
        self.assertEqual(laid_offset(BlockPos(2, 1, 0), 90), BlockPos(2, 1, 0))
        self.assertEqual(laid_offset(BlockPos(2, 1, 0), 0), BlockPos(0, 1, -2))

    def test_the_mirror_leaves_local_x_alone(self) -> None:
        # A mirror across the long edge moves nothing sideways, so a fitting
        # authored flat is in the same place either way.
        for layout in (0, 90, 180, 270):
            with self.subTest(layout=layout):
                self.assertEqual(
                    laid_offset(BlockPos(2, 1, 0), layout, mirrored=True),
                    laid_offset(BlockPos(2, 1, 0), layout),
                )

    def test_the_mirror_turns_local_depth_over(self) -> None:
        # One ahead of the path at 90 is one north; mirrored, one south.
        self.assertEqual(laid_offset(BlockPos(0, 0, -1), 90), BlockPos(0, 0, -1))
        self.assertEqual(
            laid_offset(BlockPos(0, 0, -1), 90, mirrored=True), BlockPos(0, 0, 1)
        )


class HeadingTests(unittest.TestCase):
    def test_headings_are_measured_clockwise_from_west(self) -> None:
        self.assertEqual(heading_of((-1, 0, 0)), 0.0)
        self.assertEqual(heading_of((0, 0, -1)), 90.0)
        self.assertEqual(heading_of((1, 0, 0)), 180.0)
        self.assertEqual(heading_of((0, 0, 1)), 270.0)

    def test_a_zero_gradient_has_no_heading(self) -> None:
        with self.assertRaises(ValueError):
            heading_of((0, 0, 0))

    def test_a_resolution_must_divide_a_quarter_turn(self) -> None:
        with self.assertRaises(ValueError):
            authored_headings(40.0)

    def test_only_the_first_octant_is_authored(self) -> None:
        # Rotation carries a variant into the other three quadrants and a
        # mirror carries it across the diagonal into the other half of its
        # own, so 67.5 is 22.5 and does not want authoring twice.
        self.assertEqual(authored_headings(45.0), (0.0, 45.0))
        self.assertEqual(authored_headings(22.5), (0.0, 22.5, 45.0))
        self.assertEqual(authored_headings(15.0), (0.0, 15.0, 30.0, 45.0))

    def test_a_heading_past_the_diagonal_is_not_a_variant(self) -> None:
        with self.assertRaisesRegex(ValueError, "22.5 mirrored"):
            by_authored({0.0: "a", 22.5: "b", 45.0: "c", 67.5: "d"}, 22.5)

    def test_a_heading_is_the_variant_it_is_turned_from(self) -> None:
        self.assertEqual(authored_turn(22.5, 22.5), (22.5, 0))
        self.assertEqual(authored_turn(112.5, 22.5), (22.5, 1))
        self.assertEqual(authored_turn(202.5, 22.5), (22.5, 2))
        self.assertEqual(authored_turn(292.5, 22.5), (22.5, 3))

    def test_a_gradient_snaps_to_the_nearest_authored_heading(self) -> None:
        # Thirty degrees off west: the nearest of four variants and the
        # nearest of two are not the same one.
        thirty = (-0.866, 0, -0.5)
        self.assertEqual(snap(thirty, 22.5), 22.5)
        self.assertEqual(snap(thirty, 45.0), 45.0)

    def test_every_orbit_must_be_authored(self) -> None:
        with self.assertRaises(ValueError):
            by_authored({0.0: "west"}, 22.5)

    def test_a_turned_heading_is_not_a_variant_of_its_own(self) -> None:
        with self.assertRaises(ValueError):
            by_authored({0.0: "west", 112.5: "turned"}, 22.5)


class StampingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = BlockRegistry()
        self.domains = DomainRegistry()
        self.rail = self.domains.create("rail")
        self.world = BlockWorld(self.registry)
        self.domains.bind(self.world)

    def field(self, name="minecraft:stone", position=(0, 0, 0)) -> BlockField:
        field = BlockField(self.registry.palette)
        field.set_block(position, BlockState.create(name))
        return field

    def feature(
        self,
        name="minecraft:stone",
        position=(0, 0, 0),
        elevation_mode=ElevationMode.RIGID,
        offset=None,
    ) -> PointFeature:
        return PointFeature(
            blocks=self.field(name, position),
            placement_mode=PlacementMode.OVERWRITE,
            domain=self.rail,
            offset=offset,
            elevation_mode=elevation_mode,
        )

    def point(self, position=(0, 64, 0), gradient=(-1, 0, 0), y_offset=0,
              turn=0, direction=1, layout=None):
        path = FeaturePath(rules=())
        return path.add(
            position, gradient, turn, 0, y_offset, direction, layout
        )

    def path(self, *points) -> FeaturePath:
        path = FeaturePath(rules=())
        for position, gradient in points:
            path.add(position, gradient, 0, 0, 0)
        return path

    def quadrant(self, resolution=22.5) -> dict:
        # One feature per orbit, each marking itself by the column it sits in.
        return {
            heading: self.feature(position=(index, 0, 0))
            for index, heading in enumerate(authored_headings(resolution))
        }


class FixedStampTests(StampingTestCase):
    def test_it_is_a_stamp(self) -> None:
        self.assertIsInstance(FixedStamp(self.feature()), Stamp)

    def test_every_point_gets_the_same_feature(self) -> None:
        stamp = FixedStamp(self.feature())

        self.assertIs(stamp.rule(self.point()), stamp.feature)
        self.assertIs(stamp.rule(self.point(gradient=(0, 0, -1))), stamp.feature)

    def test_it_stamps_unturned_whichever_way_the_path_runs(self) -> None:
        stamp = FixedStamp(self.feature(), self.domains)
        point = self.point(gradient=(0, 0, -1))

        [placed] = stamp.stamp(self.world, stamp.rule(point), point, clip=UNCLIPPED)
        placement = self.world.get_placement(placed)

        self.assertEqual(placement.transform.quarter_turns, 0)
        self.assertEqual(placement.transform.origin, BlockPos(0, 64, 0))


class HeadingStampTests(StampingTestCase):
    def test_it_is_a_stamp(self) -> None:
        self.assertIsInstance(HeadingStamp(self.quadrant(), 22.5), Stamp)

    def test_a_point_gets_the_feature_authored_for_its_heading(self) -> None:
        features = self.quadrant()
        stamp = HeadingStamp(features, 22.5)

        # Twenty-two and a half degrees off west.
        self.assertIs(stamp.rule(self.point(gradient=(-1, 0, -0.414))), features[22.5])
        self.assertIs(stamp.rule(self.point(gradient=(-1, 0, 0))), features[0.0])

    def test_a_turned_heading_takes_the_feature_of_its_orbit(self) -> None:
        features = self.quadrant()
        stamp = HeadingStamp(features, 22.5)
        point = self.point(gradient=(0.414, 0, -1))

        # 112.5 degrees: the 22.5 feature, one quarter round.
        self.assertIs(stamp.rule(point), features[22.5])
        self.assertEqual(stamp.at(112.5)[1], 1)
        [placed] = stamp.stamp(self.world, stamp.rule(point), point, clip=UNCLIPPED)
        placement = self.world.get_placement(placed)
        # One quarter round its orbit, laid out across travel: 0.
        self.assertEqual(placement.transform.quarter_turns, 0)

    def test_a_heading_past_the_diagonal_is_its_reflection(self) -> None:
        # 67.5 is 22.5 mirrored across the diagonal, so it is stamped from
        # the same feature, mirrored rather than turned.
        features = self.quadrant()
        stamp = HeadingStamp(features, 22.5)

        self.assertEqual(stamp.at(22.5), (features[22.5], 0, False))
        self.assertEqual(stamp.at(67.5), (features[22.5], 0, True))

    def test_the_mirror_reflects_what_it_lays_about_the_diagonal(self) -> None:
        # An L, so a reflection shows: what 67.5 lays is what 22.5 lays with
        # x and z swapped, which is what a mirror across the diagonal is.
        shape = self.feature()
        shape.blocks.set_block(BlockPos(2, 0, 0), BlockState.create("minecraft:stone"))
        shape.blocks.set_block(BlockPos(0, 0, 2), BlockState.create("minecraft:gold_block"))
        stamp = HeadingStamp({h: shape for h in (0.0, 22.5, 45.0)}, 22.5, self.domains)

        laid = {}
        for heading, gradient in ((22.5, (-1, 0, -0.414)), (67.5, (-0.414, 0, -1))):
            world = BlockWorld(BlockRegistry())
            self.domains.bind(world)
            point = self.point(gradient=gradient)
            stamp.stamp(world, stamp.rule(point), point, clip=UNCLIPPED)
            laid[heading] = {(p.x, p.z) for p in world.iter_source_positions()}

        self.assertEqual(laid[67.5], {(z, x) for x, z in laid[22.5]})

    def test_grade_does_not_change_which_feature_is_stamped(self) -> None:
        features = self.quadrant()
        stamp = HeadingStamp(features, 22.5)

        level = stamp.rule(self.point(gradient=(-1, 0, 0)))
        climbing = stamp.rule(self.point(gradient=(-1, 0.5, 0)))

        self.assertIs(level, climbing)

    def test_the_same_heading_returns_the_same_feature(self) -> None:
        # Features are shared between placements rather than rebuilt, which is
        # what keeps the world's rotated-state cache warm.
        stamp = HeadingStamp(self.quadrant(), 22.5)

        self.assertIs(stamp.at(45.0)[0], stamp.at(45.0)[0])

    def test_it_stamps_the_field_of_the_feature_it_chose(self) -> None:
        stamp = HeadingStamp(self.quadrant(), 22.5, self.domains)
        point = self.point(gradient=(-1, 0, -0.414))

        stamp.stamp(self.world, stamp.rule(point), point, clip=UNCLIPPED)

        # The 22.5 feature marks itself one column along local x, which is
        # across travel: heading 22.5 puts it a column to the north.
        self.assertEqual(
            sorted(self.world.iter_source_positions()), [BlockPos(0, 64, -1)]
        )

    def test_a_resolution_of_45_asks_for_two_features(self) -> None:
        stamp = HeadingStamp(self.quadrant(45.0), 45.0)

        self.assertEqual(stamp.headings, (0.0, 45.0))


class PlaceTests(StampingTestCase):
    def test_a_feature_offset_moves_its_origin(self) -> None:
        feature = self.feature(offset=(0, 1, 0))

        placement = self.world.get_placement(
            place(self.world, feature, self.point(), clip=UNCLIPPED, domains=self.domains)
        )

        self.assertEqual(placement.transform.origin, BlockPos(0, 65, 0))

    def test_a_rigid_feature_takes_its_own_points_elevation(self) -> None:
        feature = self.feature(elevation_mode=ElevationMode.RIGID)

        placement = self.world.get_placement(
            place(
                self.world,
                feature,
                self.point(y_offset=3),
                clip=UNCLIPPED,
                y_offset=_Raised(8),
            )
        )

        # Its point's offset, not the column elevation it was handed.
        self.assertIsNone(placement.transform.y_offset)
        self.assertEqual(placement.transform.apply((0, 0, 0)), BlockPos(0, 67, 0))

    def test_a_deformable_feature_ignores_its_points_elevation(self) -> None:
        feature = self.feature(elevation_mode=ElevationMode.DEFORMABLE)

        placement = self.world.get_placement(
            place(
                self.world,
                feature,
                self.point(y_offset=3),
                clip=UNCLIPPED,
                y_offset=_Raised(8),
            )
        )

        self.assertEqual(placement.transform.apply((0, 0, 0)), BlockPos(0, 72, 0))

    def test_a_deformable_feature_takes_the_elevation_of_its_column(self) -> None:
        feature = self.feature(elevation_mode=ElevationMode.DEFORMABLE)

        placement = self.world.get_placement(
            place(self.world, feature, self.point(), clip=UNCLIPPED, y_offset=_Raised(8))
        )

        self.assertEqual(placement.transform.apply((0, 0, 0)), BlockPos(0, 72, 0))

    def test_an_unstamped_elevation_mode_is_refused(self) -> None:
        feature = self.feature(elevation_mode=ElevationMode.INTERPOLATED)

        with self.assertRaises(NotImplementedError):
            place(self.world, feature, self.point(), clip=UNCLIPPED)

    def test_a_stamp_must_say_what_it_is_clipped_to(self) -> None:
        # Not defaulted: a path is walked one frame at a time, and the clip is
        # what keeps a shared boundary point from being placed twice.
        with self.assertRaises(TypeError):
            place(self.world, self.feature(), self.point())

    def test_unclipped_is_how_a_caller_says_it_meant_it(self) -> None:
        self.assertTrue(UNCLIPPED.contains(0, 0))
        self.assertTrue(UNCLIPPED.contains(-9999.5, 9999.5))

        place(self.world, self.feature(), self.point(), clip=UNCLIPPED)

        self.assertEqual(
            list(self.world.iter_source_positions()), [BlockPos(0, 64, 0)]
        )

    def test_a_clip_keeps_a_stamp_out_of_the_columns_outside_it(self) -> None:
        place(self.world, self.feature(), self.point(), clip=_Nowhere())

        self.assertEqual(list(self.world.iter_source_positions()), [])

    def test_a_stamp_takes_its_domain_from_the_feature(self) -> None:
        placement = self.world.get_placement(
            place(self.world, self.feature(), self.point(), clip=UNCLIPPED, domains=self.domains)
        )

        self.assertEqual(placement.connectivity_domain, self.domains.id_of(self.rail))

    def test_without_domains_a_stamp_falls_back_to_the_defaults(self) -> None:
        placement = self.world.get_placement(
            place(self.world, self.feature(), self.point(), clip=UNCLIPPED)
        )

        self.assertEqual(placement.connectivity_domain, 0)


class ClipRuleTests(StampingTestCase):
    def test_unclipped_clips_nothing(self) -> None:
        self.assertIs(unclipped(self.point()), UNCLIPPED)

    def test_in_frames_gives_a_point_the_frame_it_sits_in(self) -> None:
        left, right = _frame(0, -5.0, 0.0), _frame(1, 0.0, 5.0)
        clip = in_frames([left, right])

        self.assertIs(clip(self.point(position=(-2, 64, 0))), left)
        self.assertIs(clip(self.point(position=(2, 64, 0))), right)

    def test_a_point_in_no_frame_says_so(self) -> None:
        clip = in_frames([_frame(0, -5.0, 0.0)])

        with self.assertRaisesRegex(ValueError, r"\(9, 64, 0\) is in no frame"):
            clip(self.point(position=(9, 64, 0)))


class TurnStampTests(StampingTestCase):
    def test_it_is_a_stamp(self) -> None:
        self.assertIsInstance(TurnStamp(self.feature()), Stamp)

    def test_a_section_is_authored_across_travel_not_along_it(self) -> None:
        # West lays out at 3, north at 0, east at 1, south at 2.
        self.assertEqual([lateral_turns(turn) for turn in (0, 90, 180, 270)],
                         [3, 0, 1, 2])

    def test_it_lays_blocks_out_at_its_points_turn(self) -> None:
        stamp = TurnStamp(self.feature(), self.domains)
        point = self.point(turn=180)

        [placed] = stamp.stamp(self.world, stamp.rule(point), point, clip=UNCLIPPED)
        placement = self.world.get_placement(placed)

        self.assertEqual(placement.transform.quarter_turns, 1)

    def test_a_turn_counted_the_other_way_lays_out_the_other_way(self) -> None:
        stamp = TurnStamp(self.feature(), self.domains)
        point = self.point(turn=90, direction=-1)

        [placed] = stamp.stamp(self.world, stamp.rule(point), point, clip=UNCLIPPED)
        placement = self.world.get_placement(placed)

        # 90 counter-clockwise travels south, which lays out at 2.
        self.assertEqual(placement.transform.quarter_turns, 2)


class DiagonalTurnStampTests(StampingTestCase):
    # A diagonal is two quarter layouts staggered. The section records which
    # of them placed each position, and the stamp turns each half its own
    # way. Left to the turn alone both halves would take the quarter below
    # the diagonal, which is what tore a wedge out of the inside of a bend.

    def test_a_position_is_turned_by_the_quarter_that_placed_it(self) -> None:
        stamp = TurnStamp(self.feature(), self.domains)

        for quarter in (180, 270):
            point = self.point(turn=225, layout=quarter)
            [placed] = stamp.stamp(
                self.world, stamp.rule(point), point, clip=UNCLIPPED
            )
            self.assertEqual(
                self.world.get_placement(placed).transform.quarter_turns,
                lateral_turns(quarter),
                f"quarter {quarter}",
            )

    def test_without_one_it_falls_back_to_the_points_own_turn(self) -> None:
        # A lane paints onto the path rather than being placed in a layout.
        stamp = TurnStamp(self.feature(), self.domains)
        point = self.point(turn=180)

        self.assertIsNone(point.layout)
        self.assertEqual(stamp.layout(point), 180)

    def test_the_two_halves_of_a_diagonal_take_different_layouts(self) -> None:
        stamp = TurnStamp(self.feature(), self.domains)

        self.assertNotEqual(
            stamp.layout(self.point(turn=225, layout=180)),
            stamp.layout(self.point(turn=225, layout=270)),
        )


class StampAlongTests(StampingTestCase):
    def test_every_point_of_a_feature_path_is_stamped(self) -> None:
        stamp = FixedStamp(self.feature(), self.domains)
        path = self.path(((0, 64, 0), (-1, 0, 0)), ((1, 64, 0), (-1, 0, 0)))

        placements = stamp_along(stamp, self.world, path, clip=unclipped)

        self.assertEqual(len(placements), 2)
        self.assertEqual(
            sorted(self.world.iter_source_positions()),
            [BlockPos(0, 64, 0), BlockPos(1, 64, 0)],
        )

    def test_a_path_that_turns_changes_feature_part_way_through(self) -> None:
        features = self.quadrant()
        stamp = HeadingStamp(features, 22.5, self.domains)
        path = self.path(((0, 64, 0), (-1, 0, 0)), ((10, 64, 0), (-1, 0, -0.414)))

        stamp_along(stamp, self.world, path, clip=unclipped)

        # The 0 feature marks its own column; the 22.5 one marks the column
        # beside it, since local x is across travel rather than along it.
        self.assertEqual(
            sorted(self.world.iter_source_positions()),
            [BlockPos(0, 64, 0), BlockPos(10, 64, -1)],
        )

    def test_an_empty_path_stamps_nothing(self) -> None:
        stamp = FixedStamp(self.feature(), self.domains)

        self.assertEqual(stamp_along(stamp, self.world, self.path(), clip=unclipped), ())


def _frame(index: int, start: float, end: float) -> PathFrame:
    # A frame holding the columns from `start` up to `end` along x.
    return PathFrame(
        index=index,
        path_start=index,
        path_stop=index + 1,
        start_boundary=Boundary.from_direction((start, 0.0), (1.0, 0.0)),
        end_boundary=Boundary.from_direction((end, 0.0), (1.0, 0.0)),
        y_offset=0,
    )


class _Raised:
    # A YOffset that lifts every column by the same amount.

    def __init__(self, height: int) -> None:
        self.height = height

    def __call__(self, x: int, z: int) -> int:
        return self.height

    @property
    def span(self) -> tuple[int, int]:
        return self.height, self.height


class _Nowhere(ColumnMask):
    def contains(self, x: float, z: float) -> bool:
        return False


if __name__ == "__main__":
    unittest.main()
