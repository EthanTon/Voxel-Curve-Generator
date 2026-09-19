import unittest

from src.Core.Domain import Domain
from src.Core.Feature.feature import PointFeature
from src.Core.Field.block_field import BlockField
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos
from src.Core.modes import PlacementMode, StepMode
from src.Format.markers import (
    ANCHOR,
    ANNOTATION,
    FEATURE,
    SEPARATOR,
    anchor,
    block_field,
    markers,
    of_kind,
    parse_tags,
    rebased,
    section_from_region,
)
from src.Format.region import BlockRegion, RegionEntity

DISPLAY = "minecraft:block_display"


def section(*entities):
    region = BlockRegion("section", (0, 0, 0), (5, 3, 3))
    for entity in entities:
        region.add_entity(entity)
    return region


def marker_entity(pos, *tags):
    return RegionEntity(DISPLAY, pos, tags)


class TagTests(unittest.TestCase):
    def test_a_bare_tag_names_the_kind_and_the_rest_are_properties(self) -> None:
        kind, properties = parse_tags(
            ["vcg.annotation", "vcg.name-left_rail", "vcg.step-chebyshev"]
        )

        self.assertEqual(SEPARATOR, "-")
        self.assertEqual(kind, ANNOTATION)
        self.assertEqual(properties, {"name": "left_rail", "step": "chebyshev"})

    def test_only_the_first_separator_splits_so_a_value_may_contain_it(self) -> None:
        _, properties = parse_tags(["vcg.feature-egg-egg", "vcg.name-a-b-c"])

        self.assertEqual(properties, {"feature": "egg-egg", "name": "a-b-c"})

    def test_the_separator_is_configurable(self) -> None:
        self.assertEqual(
            parse_tags(["vcg.annotation", "vcg.name=left_rail"], separator="="),
            (ANNOTATION, {"name": "left_rail"}),
        )

    def test_a_property_with_the_wrong_separator_is_refused(self) -> None:
        # Otherwise it silently becomes a kind called "name=left_rail".
        with self.assertRaises(ValueError) as caught:
            parse_tags(["vcg.name=left_rail"])

        self.assertIn("neither a kind nor", str(caught.exception))

    def test_tags_outside_the_namespace_are_ignored(self) -> None:
        kind, properties = parse_tags(["mypack.debug", "smithed.entity", "vcg.feature"])

        self.assertEqual(kind, FEATURE)
        self.assertEqual(properties, {})

    def test_an_entity_with_no_namespaced_tag_is_not_a_marker(self) -> None:
        self.assertEqual(parse_tags(["mypack.debug"]), (None, {}))
        self.assertEqual(markers(section(marker_entity((0.5, 0.5, 0.5), "mypack.debug"))), [])

    def test_two_kinds_on_one_entity_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            parse_tags(["vcg.annotation", "vcg.feature"])

    def test_a_value_may_contain_a_colon_or_slash(self) -> None:
        _, properties = parse_tags(["vcg.block-minecraft:rail", "vcg.path-a/b"])

        self.assertEqual(properties, {"block": "minecraft:rail", "path": "a/b"})

    def test_the_namespace_is_configurable(self) -> None:
        kind, properties = parse_tags(["rail.annotation", "rail.name-x"], namespace="rail")

        self.assertEqual((kind, properties), (ANNOTATION, {"name": "x"}))

    def test_markers_thread_the_separator_through(self) -> None:
        region = section(marker_entity((0.0, 0.0, 0.0), "vcg.annotation", "vcg.name=a"))

        self.assertEqual(markers(region, separator="=")[0].name, "a")


class MarkerTests(unittest.TestCase):
    def test_a_marker_takes_the_floored_position_of_its_entity(self) -> None:
        region = section(marker_entity((2.5, 0.9, -1.2), "vcg.annotation", "vcg.name-a"))

        self.assertEqual(markers(region)[0].position, BlockPos(2, 0, -2))

    def test_step_mode_defaults_to_manhattan(self) -> None:
        region = section(
            marker_entity((0.0, 0.0, 0.0), "vcg.annotation", "vcg.name-a"),
            marker_entity((1.0, 0.0, 0.0), "vcg.annotation", "vcg.name-b", "vcg.step-chebyshev"),
        )

        found = markers(region)
        self.assertEqual(found[0].step_mode, StepMode.MANHATTAN)
        self.assertEqual(found[1].step_mode, StepMode.CHEBYSHEV)

    def test_a_missing_required_tag_names_the_marker(self) -> None:
        region = section(marker_entity((2.0, 1.0, 0.0), "vcg.annotation"))

        with self.assertRaises(KeyError) as caught:
            markers(region)[0].name

        self.assertIn("(2, 1, 0)", str(caught.exception))

    def test_of_kind_separates_annotations_from_features(self) -> None:
        region = section(
            marker_entity((0.0, 0.0, 0.0), "vcg.annotation", "vcg.name-a"),
            marker_entity((1.0, 0.0, 0.0), "vcg.feature", "vcg.name-rail"),
        )

        found = markers(region)
        self.assertEqual([m.name for m in of_kind(found, ANNOTATION)], ["a"])
        self.assertEqual([m.name for m in of_kind(found, FEATURE)], ["rail"])

    def test_the_anchor_rebases_every_other_marker(self) -> None:
        # Section-local coordinates are relative to the path point.
        region = section(
            marker_entity((2.5, 0.5, 1.5), f"vcg.{ANCHOR}"),
            marker_entity((1.5, 1.5, 1.5), "vcg.annotation", "vcg.name-left"),
            marker_entity((3.5, 1.5, 1.5), "vcg.annotation", "vcg.name-right"),
        )

        found = rebased(markers(region))

        self.assertEqual(
            [(m.name, tuple(m.position)) for m in found],
            [("left", (-1, 1, 0)), ("right", (1, 1, 0))],
        )

    def test_exactly_one_anchor_is_required(self) -> None:
        none = section(marker_entity((0.0, 0.0, 0.0), "vcg.annotation", "vcg.name-a"))
        two = section(
            marker_entity((0.0, 0.0, 0.0), f"vcg.{ANCHOR}"),
            marker_entity((1.0, 0.0, 0.0), f"vcg.{ANCHOR}"),
        )

        for region in (none, two):
            with self.assertRaises(ValueError):
                anchor(markers(region))


class SectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = {"egg_egg": self.feature(), "egg-egg": self.feature()}

    def feature(self):
        return PointFeature(BlockField(), PlacementMode.OVERWRITE, Domain(1, "test"))

    def region(self, *extra):
        region = BlockRegion("main", (128, 70, -44), (5, 3, 1))
        for x in range(128, 133):
            region.set_state((x, 70, -44), BlockState.create("minecraft:stone"))
        region.add_entity(marker_entity((130.5, 70.5, -43.5), f"vcg.{ANCHOR}"))
        region.add_entity(
            marker_entity(
                (129.5, 71.5, -43.5),
                "vcg.annotation",
                "vcg.name-123123",
                "vcg.feature-egg_egg",
            )
        )
        for entity in extra:
            region.add_entity(entity)
        return region

    def test_annotations_keep_region_local_positions(self) -> None:
        section = section_from_region(self.region(), self.features)

        annotation = next(iter(section.points))
        self.assertEqual(annotation.name, "123123")
        self.assertEqual(annotation.position, BlockPos(1, 1, 0))

    def test_the_anchor_becomes_the_section_origin(self) -> None:
        # CrossSection subtracts origin, so the anchor goes there rather than
        # being baked into every position.
        section = section_from_region(self.region(), self.features)

        self.assertEqual(section.origin, BlockPos(2, 0, 0))

    def test_the_feature_is_the_caller_supplied_object(self) -> None:
        section = section_from_region(self.region(), self.features)

        self.assertIs(next(iter(section.points)).feature, self.features["egg_egg"])

    def test_step_mode_comes_from_the_tag(self) -> None:
        region = self.region(
            marker_entity(
                (131.5, 71.5, -43.5),
                "vcg.annotation",
                "vcg.name-456456",
                "vcg.feature-egg-egg",
                "vcg.step-chebyshev",
            )
        )

        modes = {a.name: a.step_mode for a in section_from_region(region, self.features).points}
        self.assertEqual(modes, {"123123": StepMode.MANHATTAN, "456456": StepMode.CHEBYSHEV})

    def test_an_unknown_feature_names_what_was_supplied(self) -> None:
        region = self.region(
            marker_entity((131.5, 71.5, -43.5), "vcg.annotation", "vcg.name-x", "vcg.feature-missing")
        )

        with self.assertRaises(KeyError) as caught:
            section_from_region(region, self.features)

        self.assertIn("missing", str(caught.exception))
        self.assertIn("egg_egg", str(caught.exception))

    def test_duplicate_annotation_names_are_refused(self) -> None:
        region = self.region(
            marker_entity((131.5, 71.5, -43.5), "vcg.annotation", "vcg.name-123123", "vcg.feature-egg_egg")
        )

        with self.assertRaises(ValueError):
            section_from_region(region, self.features)

    def test_an_annotation_without_a_feature_tag_is_refused(self) -> None:
        region = self.region(
            marker_entity((131.5, 71.5, -43.5), "vcg.annotation", "vcg.name-x")
        )

        with self.assertRaises(KeyError):
            section_from_region(region, self.features)

    def test_the_section_is_usable_not_just_populated(self) -> None:
        section = section_from_region(self.region(), self.features)

        self.assertGreater(section.reach(), 0.0)
        self.assertEqual(len(section.orient(0)), 1)

    def test_block_field_carries_the_authored_blocks_locally(self) -> None:
        field = block_field(self.region())

        self.assertEqual(len(field), 5)
        self.assertEqual(field.bounds.minimum, BlockPos(0, 0, 0))
        self.assertEqual(field.bounds.maximum, BlockPos(4, 0, 0))

    def test_block_field_can_be_based_on_the_anchor(self) -> None:
        region = self.region()
        origin = anchor(markers(region)).position

        field = block_field(region, origin=origin)

        self.assertEqual(field.bounds.minimum, BlockPos(-2, 0, 0))


if __name__ == "__main__":
    unittest.main()
