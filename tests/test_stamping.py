import unittest

from src.Core.Domain import DomainRegistry
from src.Core.Feature.feature import PointFeature
from src.Core.Field.block_field import BlockField
from src.Core.Feature.feature_path import FeaturePath
from src.Core.World.block_world import BlockWorld, ColumnMask
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos, Vec3
from src.Core.modes import ElevationMode, PlacementMode
from src.Stamping import (
    UNCLIPPED,
    FixedStamp,
    HeadingStamp,
    Stamp,
    authored_headings,
    authored_turn,
    by_authored,
    heading_of,
    place,
    snap,
    stamp_along,
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

    def test_one_heading_authored_per_rotation(self) -> None:
        self.assertEqual(authored_headings(45.0), (0.0, 45.0))
        self.assertEqual(authored_headings(22.5), (0.0, 22.5, 45.0, 67.5))

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

    def point(self, position=(0, 64, 0), gradient=(-1, 0, 0), y_offset=0):
        path = FeaturePath(rules=())
        return path.add(position, gradient, 0, 0, y_offset)

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

        placement = self.world.get_placement(
            stamp.stamp(self.world, stamp.rule(point), point, clip=UNCLIPPED)
        )

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
        placement = self.world.get_placement(
            stamp.stamp(self.world, stamp.rule(point), point, clip=UNCLIPPED)
        )
        self.assertEqual(placement.transform.quarter_turns, 1)

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

        # The 22.5 feature marks itself one column right of the origin.
        self.assertEqual(
            sorted(self.world.iter_source_positions()), [BlockPos(1, 64, 0)]
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


class StampAlongTests(StampingTestCase):
    def test_every_point_of_a_feature_path_is_stamped(self) -> None:
        stamp = FixedStamp(self.feature(), self.domains)
        path = self.path(((0, 64, 0), (-1, 0, 0)), ((1, 64, 0), (-1, 0, 0)))

        placements = stamp_along(stamp, self.world, path, clip=UNCLIPPED)

        self.assertEqual(len(placements), 2)
        self.assertEqual(
            sorted(self.world.iter_source_positions()),
            [BlockPos(0, 64, 0), BlockPos(1, 64, 0)],
        )

    def test_a_path_that_turns_changes_feature_part_way_through(self) -> None:
        features = self.quadrant()
        stamp = HeadingStamp(features, 22.5, self.domains)
        path = self.path(((0, 64, 0), (-1, 0, 0)), ((10, 64, 0), (-1, 0, -0.414)))

        stamp_along(stamp, self.world, path, clip=UNCLIPPED)

        # The 0 feature marks its own column; the 22.5 one marks the next.
        self.assertEqual(
            sorted(self.world.iter_source_positions()),
            [BlockPos(0, 64, 0), BlockPos(11, 64, 0)],
        )

    def test_an_empty_path_stamps_nothing(self) -> None:
        stamp = FixedStamp(self.feature(), self.domains)

        self.assertEqual(stamp_along(stamp, self.world, self.path(), clip=UNCLIPPED), ())


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
