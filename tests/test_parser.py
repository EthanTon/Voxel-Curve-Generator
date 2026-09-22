import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.Assembly.structure import build_domains, build_path
from src.Core.block.connectivity_options import ConnectionAction
from src.Core.coordinates import BlockPos, ColumnPos
from src.Core.modes import PlacementMode, TerrainMaskMode
from src.Parser import load, parse


def config(**overrides) -> dict:
    data = {
        "path": {
            "points": [[0, 64, 0], [1, 64, 0], [2, 64, 0]],
            "gradients": [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
        },
        "output": {"path": "out.litematic"},
        "domain": [{"name": "rail"}],
    }
    data.update(overrides)
    return data


class ConfigTests(unittest.TestCase):
    def test_a_config_becomes_a_spec(self) -> None:
        spec = parse(config())

        self.assertEqual(spec.path.points[0], BlockPos(0, 64, 0))
        self.assertEqual(spec.output.name, "build")
        self.assertEqual(spec.domain("rail").name, "rail")

    def test_the_output_is_written_beside_the_build(self) -> None:
        # Every other path in a build is relative to the build's own file,
        # and the output is too, so it does not move with the working
        # directory the build was run from.
        spec = parse(config(), Path("builds/curve"))

        self.assertEqual(spec.output.path, Path("builds/curve/out.litematic"))

    def test_an_absolute_output_is_left_where_it_was_written(self) -> None:
        data = config(output={"path": "/tmp/out.litematic"})

        spec = parse(data, Path("builds/curve"))

        self.assertEqual(spec.output.path, Path("/tmp/out.litematic"))

    def test_a_path_lays_the_join_close_to_the_diagonal_by_default(self) -> None:
        self.assertEqual(parse(config()).path.diagonal_degrees, 2.0)

    def test_a_path_may_narrow_what_counts_as_a_diagonal(self) -> None:
        data = config()
        data["path"]["diagonal_degrees"] = 1.0

        self.assertEqual(parse(data).path.diagonal_degrees, 1.0)

    def test_a_diagonal_wider_than_half_a_step_is_refused(self) -> None:
        data = config()
        data["path"]["diagonal_degrees"] = 60

        with self.assertRaisesRegex(ValueError, "between 0 and 45 degrees"):
            parse(data)

    def test_a_missing_key_says_which_one(self) -> None:
        data = config()
        del data["path"]["gradients"]

        with self.assertRaisesRegex(ValueError, "gradients"):
            parse(data)

    def test_an_unknown_mode_lists_the_known_ones(self) -> None:
        data = config()
        data["domain"][0]["placement"] = {"mode": "squishy"}

        with self.assertRaisesRegex(ValueError, "KEEP_EXISTING"):
            parse(data)

    def test_an_unknown_domain_reference_is_refused(self) -> None:
        data = config(marker={"a": [0, 0], "b": [2, 0]})
        data["section"] = [
            {"name": "deck", "domain": "missing", "schematic": "d.schem",
             "spans": [["a", "b"]]}
        ]

        with self.assertRaisesRegex(KeyError, "No domain named 'missing'"):
            parse(data)



class DomainTests(unittest.TestCase):
    def domained(self, *domains, interactions=()) -> dict:
        return parse(config(domain=list(domains), interaction=list(interactions)))

    def test_every_rule_a_domain_has_can_be_configured(self) -> None:
        spec = self.domained(
            {
                "name": "structure",
                "properties": {"material": "concrete", "bearing": 40},
                "placement": {
                    "mode": "KEEP_EXISTING",
                    "replace": "minecraft:dirt",
                    "protected": True,
                },
                "connectivity": {"connects_to_self": True, "connects_to_any": False},
                "terrain": {"is_terrain": False, "mask_mode": "SURFACE"},
            }
        )

        domain = spec.domain("structure")
        self.assertEqual(domain.properties, {"material": "concrete", "bearing": 40})
        self.assertEqual(domain.placement.mode, PlacementMode.KEEP_EXISTING)
        self.assertEqual(domain.placement.replace, "minecraft:dirt")
        self.assertTrue(domain.placement.protected)
        self.assertTrue(domain.connectivity.connects_to_self)
        self.assertEqual(domain.terrain.mask_mode, TerrainMaskMode.SURFACE)

    def test_an_unset_rule_is_left_to_be_inherited(self) -> None:
        spec = self.domained({"name": "bare"})

        domain = spec.domain("bare")
        self.assertIsNone(domain.placement.mode)
        self.assertIsNone(domain.connectivity.connects_to_self)
        self.assertIsNone(domain.terrain.is_terrain)

    def test_rules_reach_the_compiled_profile(self) -> None:
        spec = self.domained(
            {"name": "structure", "placement": {"protected": True}},
            {"name": "rail", "parents": ["structure"],
             "placement": {"mode": "OVERWRITE"}},
        )

        profile = build_domains(spec).profile("rail")

        self.assertTrue(profile.protected)
        self.assertEqual(profile.placement_mode, PlacementMode.OVERWRITE)

    def test_an_unknown_mask_mode_lists_the_known_ones(self) -> None:
        with self.assertRaisesRegex(ValueError, "FOLLOW_SURFACE"):
            self.domained({"name": "a", "terrain": {"mask_mode": "sideways"}})

    def test_an_interaction_names_two_domains(self) -> None:
        spec = self.domained(
            {"name": "rail"},
            {"name": "structure"},
            interactions=[
                {"source": "rail", "target": "structure", "connect": True,
                 "grants": {"supported": True}},
            ],
        )

        self.assertEqual(spec.interactions[0].source, "rail")
        self.assertTrue(spec.interactions[0].symmetric)

    def test_an_interaction_grants_reach_the_profile(self) -> None:
        spec = self.domained(
            {"name": "rail"},
            {"name": "structure"},
            interactions=[
                {"source": "rail", "target": "structure",
                 "grants": {"supported": True}, "symmetric": False},
            ],
        )

        domains = build_domains(spec)

        self.assertEqual(domains.profile("rail").properties["supported"], True)
        self.assertNotIn("supported", domains.profile("structure").properties)

    def test_overwriting_is_directional(self) -> None:
        # Both protected, so only the interaction lets anything through.
        spec = self.domained(
            {"name": "structure", "placement": {"protected": True}},
            {"name": "rail", "placement": {"protected": True}},
            interactions=[
                {"source": "rail", "target": "structure", "overwrite": True,
                 "symmetric": False},
            ],
        )

        domains = build_domains(spec)
        policy = domains.compile()

        self.assertTrue(
            policy.placement_rule(domains.id_of("rail"), domains.id_of("structure"))
        )
        self.assertFalse(
            policy.placement_rule(domains.id_of("structure"), domains.id_of("rail"))
        )

    def test_an_interaction_must_name_domains_that_exist(self) -> None:
        with self.assertRaisesRegex(KeyError, "No domain named 'nowhere'"):
            self.domained(
                {"name": "rail"},
                interactions=[{"source": "rail", "target": "nowhere"}],
            )

    def test_a_connection_override_is_configured_in_full(self) -> None:
        spec = self.domained(
            {"name": "rail"},
            {"name": "glass"},
            interactions=[
                {
                    "source": "rail",
                    "target": "glass",
                    "override": [
                        {
                            "action": "disconnect",
                            "source_blocks": ["minecraft:rail"],
                            "neighbor_blocks": ["minecraft:glass"],
                            "directions": ["north", "south"],
                            "source_properties": {"shape": ["north_south"]},
                            "symmetric": True,
                        }
                    ],
                }
            ],
        )

        override = spec.interactions[0].overrides[0]
        self.assertEqual(override.action, ConnectionAction.DISCONNECT)
        self.assertEqual(override.source_blocks, frozenset({"minecraft:rail"}))
        self.assertEqual(override.directions, frozenset({"north", "south"}))
        self.assertEqual(override.source_properties, {"shape": frozenset({"north_south"})})
        self.assertTrue(override.symmetric)

    def test_a_domain_may_carry_its_own_overrides(self) -> None:
        spec = self.domained(
            {
                "name": "rail",
                "connectivity": {
                    "override": [{"action": "connect",
                                  "neighbor_blocks": ["minecraft:stone"]}]
                },
            }
        )

        self.assertEqual(len(spec.domain("rail").connectivity.overrides), 1)

    def test_an_override_may_not_scope_itself(self) -> None:
        # Where it is declared is what scopes it.
        with self.assertRaisesRegex(ValueError, "taken from where it is declared"):
            self.domained(
                {"name": "rail",
                 "connectivity": {"override": [{"source_domains": ["rail"]}]}}
            )



class MarkerTests(unittest.TestCase):
    def test_markers_are_named_columns(self) -> None:
        spec = parse(config(marker={"start": [-22998, -5870], "end": [-22672, -8097]}))

        self.assertEqual(spec.marker("start"), ColumnPos(-22998, -5870))
        self.assertEqual(len(spec.markers), 2)

    def test_a_marker_needs_an_x_and_a_z(self) -> None:
        with self.assertRaisesRegex(ValueError, "marker 'start' needs an x and a z"):
            parse(config(marker={"start": [1, 2, 3]}))

    def test_an_unknown_marker_lists_the_known_ones(self) -> None:
        spec = parse(config(marker={"start": [0, 0]}))

        with self.assertRaisesRegex(KeyError, "No marker named 'middle'"):
            spec.marker("middle")



class PathTests(unittest.TestCase):
    def test_a_path_is_taken_as_given(self) -> None:
        spec = parse(config()).path

        self.assertEqual(spec.points[0], BlockPos(0, 64, 0))
        self.assertEqual(spec.lead, 0)

    def test_points_and_gradients_must_match(self) -> None:
        data = config()
        data["path"]["gradients"] = [[1, 0, 0]]

        with self.assertRaisesRegex(ValueError, "3 points and 1 gradients"):
            parse(data)

    def test_a_lead_is_carried_onto_the_path(self) -> None:
        data = config()
        data["path"]["lead"] = 1

        self.assertEqual(parse(data).path.lead, 1)

    def test_a_lead_must_be_non_negative(self) -> None:
        data = config()
        data["path"]["lead"] = -1

        with self.assertRaisesRegex(ValueError, "lead must be non-negative"):
            parse(data)

    def test_a_lead_needs_points_to_lead_from(self) -> None:
        data = config()
        data["path"]["lead"] = 2

        with self.assertRaisesRegex(ValueError, "needs more than 4 points"):
            parse(data)

    def test_a_lead_extends_the_ends(self) -> None:
        data = config()
        data["path"]["lead"] = 1

        path = build_path(parse(data))

        self.assertEqual(len(path.path_points), 5)
        self.assertEqual(path.first_index, 1)
        self.assertEqual(path.last_index, 3)



class PathFileTests(unittest.TestCase):
    def build(self, directory: Path, contents: str) -> dict:
        (directory / "lamvel_path.toml").write_text(contents)
        (directory / "build.toml").write_text(
            '[path]\npath = "lamvel_path.toml"\n'
            '[output]\npath = "out.litematic"\n'
        )
        return load(directory / "build.toml")

    def test_a_path_can_come_from_its_own_file(self) -> None:
        with TemporaryDirectory() as directory:
            spec = self.build(
                Path(directory),
                "points = [[0, 64, 0], [1, 64, 0]]\n"
                "gradients = [[1, 0, 0], [1, 0, 0]]\n",
            )

        self.assertEqual(spec.path.points, (BlockPos(0, 64, 0), BlockPos(1, 64, 0)))

    def test_a_path_file_may_wrap_its_points_in_a_path_table(self) -> None:
        with TemporaryDirectory() as directory:
            spec = self.build(
                Path(directory),
                "[path]\npoints = [[0, 64, 0], [1, 64, 0]]\n"
                "gradients = [[1, 0, 0], [1, 0, 0]]\n",
            )

        self.assertEqual(len(spec.path.points), 2)

    def test_a_path_file_is_found_next_to_the_build_that_names_it(self) -> None:
        # Not next to whatever directory the command was run from.
        with TemporaryDirectory() as directory:
            spec = self.build(
                Path(directory),
                "points = [[0, 64, 0], [1, 64, 0]]\n"
                "gradients = [[1, 0, 0], [1, 0, 0]]\nlead = 0\n",
            )

        self.assertEqual(spec.path.lead, 0)

    def test_a_missing_path_file_says_where_it_looked(self) -> None:
        with TemporaryDirectory() as directory:
            build = Path(directory) / "build.toml"
            build.write_text(
                '[path]\npath = "nowhere.toml"\n'
                '[output]\npath = "out.litematic"\n'
            )

            with self.assertRaisesRegex(ValueError, "no path file at .*nowhere"):
                load(build)



class SectionFileTests(unittest.TestCase):
    def build(self, directory: Path, listing: str, **files) -> dict:
        (directory / "sections").mkdir(parents=True, exist_ok=True)
        (directory / "sections" / "assignments.toml").write_text(listing)
        for name, contents in files.items():
            (directory / "sections" / f"{name}.toml").write_text(contents)
        (directory / "build.toml").write_text(
            'sections = "sections/assignments.toml"\n'
            "[marker]\nstart = [0, 0]\nend = [2, 0]\n"
            "[path]\npoints = [[0, 64, 0], [1, 64, 0], [2, 64, 0]]\n"
            "gradients = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]\n"
            '[output]\npath = "out.litematic"\n'
            '[[domain]]\nname = "rail"\n'
        )
        return load(directory / "build.toml")

    def test_assignments_may_live_in_a_file_of_their_own(self) -> None:
        with TemporaryDirectory() as directory:
            spec = self.build(
                Path(directory),
                '[[section]]\nname = "deck"\ndomain = "rail"\n'
                'schematic = "deck.schem"\nspans = [["start", "end"]]\n',
            )

        self.assertEqual([section.name for section in spec.sections], ["deck"])

    def test_what_that_file_names_is_found_beside_it(self) -> None:
        # Not beside the build, which may be a directory up.
        with TemporaryDirectory() as directory:
            spec = self.build(
                Path(directory),
                '[[section]]\nname = "deck"\ndomain = "rail"\n'
                'schematic = "deck.schem"\nspans = [["start", "end"]]\n',
            )

        self.assertEqual(
            spec.section("deck").schematic,
            Path(directory) / "sections" / "deck.schem",
        )

    def test_a_definition_may_live_in_a_file_of_its_own(self) -> None:
        with TemporaryDirectory() as directory:
            spec = self.build(
                Path(directory),
                '[[section]]\nname = "deck"\nfile = "deck.toml"\n'
                'spans = [["start", "end"]]\n',
                deck='domain = "rail"\nschematic = "deck.schem"\n'
                     "width = 5\ndepth = 1\n",
            )

        section = spec.section("deck")
        self.assertEqual(section.domain, "rail")
        self.assertEqual(section.width, 5)

    def test_a_definition_may_wrap_itself_in_a_section_table(self) -> None:
        with TemporaryDirectory() as directory:
            spec = self.build(
                Path(directory),
                '[[section]]\nname = "deck"\nfile = "deck.toml"\n'
                'spans = [["start", "end"]]\n',
                deck='[section]\ndomain = "rail"\nschematic = "deck.schem"\n',
            )

        self.assertEqual(spec.section("deck").domain, "rail")

    def test_the_assignment_wins_over_the_definition(self) -> None:
        with TemporaryDirectory() as directory:
            spec = self.build(
                Path(directory),
                '[[section]]\nname = "deck"\nfile = "deck.toml"\n'
                'width = 9\nspans = [["start", "end"]]\n',
                deck='domain = "rail"\nschematic = "deck.schem"\nwidth = 5\n',
            )

        self.assertEqual(spec.section("deck").width, 9)

    def test_a_missing_definition_says_where_it_looked(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no section file at .*gone"):
                self.build(
                    Path(directory),
                    '[[section]]\nname = "deck"\nfile = "gone.toml"\n'
                    'spans = [["start", "end"]]\n',
                )

    def test_sections_are_listed_in_one_place_or_the_other(self) -> None:
        data = config(
            marker={"a": [0, 0], "b": [2, 0]},
            sections="sections.toml",
            section=[{"name": "deck", "domain": "rail",
                      "schematic": "d.schem", "spans": [["a", "b"]]}],
        )

        with self.assertRaisesRegex(ValueError, "not both"):
            parse(data)


class BuildOrderTests(unittest.TestCase):
    def test_a_domain_says_when_its_blocks_go_in(self) -> None:
        spec = parse(config(domain=[{"name": "paint", "order": 10}]))

        self.assertEqual(spec.domain("paint").order, 10)

    def test_a_domain_goes_in_first_unless_it_says_otherwise(self) -> None:
        self.assertEqual(parse(config()).domain("rail").order, 0)


class PlacementModeTests(unittest.TestCase):
    def test_a_fitting_may_say_how_it_is_placed(self) -> None:
        data = config(
            marker={"a": [0, 0], "b": [2, 0]},
            section=[{"name": "deck", "domain": "rail", "schematic": "d.schem",
                      "spans": [["a", "b"]],
                      "placement_mode": "REPLACE_AIR_ONLY"}],
        )

        self.assertEqual(
            parse(data).section("deck").placement_mode,
            PlacementMode.REPLACE_AIR_ONLY,
        )

    def test_saying_nothing_leaves_it_to_the_domain(self) -> None:
        data = config(
            marker={"a": [0, 0], "b": [2, 0]},
            section=[{"name": "deck", "domain": "rail", "schematic": "d.schem",
                      "spans": [["a", "b"]]}],
        )

        self.assertIsNone(parse(data).section("deck").placement_mode)


class AssignmentFileTests(unittest.TestCase):
    def build(self, directory: Path, assignments: str) -> dict:
        (directory / "sections").mkdir(parents=True, exist_ok=True)
        (directory / "sections" / "assignments.toml").write_text(assignments)
        (directory / "build.toml").write_text(
            'path_assignments = "sections/assignments.toml"\n'
            "[marker]\nstart = [0, 0]\nend = [2, 0]\n"
            "[path]\npoints = [[0, 64, 0], [1, 64, 0], [2, 64, 0]]\n"
            "gradients = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]\n"
            '[output]\npath = "out.litematic"\n'
            '[[domain]]\nname = "rail"\n'
            '[[section]]\nname = "deck"\ndomain = "rail"\n'
            'schematic = "deck.schem"\n'
        )
        return load(directory / "build.toml")

    def test_assignments_may_live_in_a_file_of_their_own(self) -> None:
        with TemporaryDirectory() as directory:
            spec = self.build(
                Path(directory),
                '[[assignment]]\nfrom = "start"\nto = "end"\n'
                'sections = ["deck"]\n',
            )

        self.assertEqual(spec.assignments[0].sections, ("deck",))
        self.assertEqual(str(spec.assignments[0].span), "start..end")

    def test_a_missing_assignment_file_says_where_it_looked(self) -> None:
        with TemporaryDirectory() as directory:
            build = Path(directory) / "build.toml"
            build.write_text(
                'path_assignments = "gone.toml"\n'
                "[path]\npoints = [[0, 64, 0]]\ngradients = [[1, 0, 0]]\n"
                '[output]\npath = "out.litematic"\n'
            )

            with self.assertRaisesRegex(
                ValueError, "no path assignments file at .*gone"
            ):
                load(build)

    def test_assignments_are_listed_in_one_place_or_the_other(self) -> None:
        data = config(
            marker={"a": [0, 0], "b": [2, 0]},
            path_assignments="elsewhere.toml",
            assignment=[{"from": "a", "to": "b", "sections": ["deck"]}],
        )

        with self.assertRaisesRegex(ValueError, "not both"):
            parse(data)

    def test_a_build_may_use_none_of_the_sections_it_defines(self) -> None:
        # A definitions file is a library; a build takes what it needs.
        data = config(
            marker={"a": [0, 0], "b": [2, 0]},
            section=[
                {"name": "deck", "domain": "rail", "schematic": "d.schem"},
                {"name": "spare", "domain": "rail", "schematic": "s.schem"},
            ],
            assignment=[{"from": "a", "to": "b", "sections": ["deck"]}],
        )

        spec = parse(data)

        self.assertEqual(len(spec.sections), 2)
        self.assertEqual(spec.assignments[0].sections, ("deck",))


if __name__ == "__main__":
    unittest.main()
