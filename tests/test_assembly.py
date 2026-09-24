import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.Assembly import assemble, write
from src.Parser import load, parse
from src.Assembly.anchors import fitted_sections, points_for, runs_under
from src.Assembly.assemble import build as build_assembly, sweep
from src.Assembly.lanes import lane_support, mark_lane
from src.Assembly.wires import strand_of
from src.Assembly.structure import (
    build_domains,
    build_elevation_frames,
    build_path_frames,
    build_path,
    build_schedules,
    build_sections,
    build_structure,
    resolve_spans,
)
from src.Parser.spec import SectionElevation, WireGeometry
from src.Core.coordinates import BlockPos, ColumnPos
from src.Core.line import line
from src.Core.block.connectivity_options import ConnectionAction
from src.Core.modes import (
    ElevationMode,
    LineMode,
    PlacementMode,
    StepMode,
    TerrainMaskMode,
)
from src.Core.Feature.feature_path import FeaturePath
from src.Core.Field.block_field import BlockField
from src.Core.block.registry import BlockRegistry
from src.Core.block.states import BlockState
from src.Format.region import BlockRegion
from src.Format import nbt
from src.Format.litematic import load_litematic
from src.Format.schematics import load_schematic, state_key, write_schematic
from src.Generation.Geometry.frames import level_near
from src.Generation.Sweep.delivery import cover
from src.Stamping.stamp import laid_offset


class CurveRailRegressionTests(unittest.TestCase):
    def test_rails_follow_the_sleeper_columns_through_climbs_and_join(self):
        sample = Path(__file__).resolve().parents[1] / "sandbox/curve800/build.toml"
        spec = load(sample)
        path = build_path(spec)
        spans = resolve_spans(spec, path)
        sections = build_sections(spec, BlockRegistry(), build_domains(spec), spans)
        schedules = build_schedules(sections, spans, len(path.path_points))
        frames = build_path_frames(spec, schedules, path)
        runs = sweep(spec, schedules, path, frames)
        centre = next(run for run in runs if run.annotation.name == "trackbed")
        level = level_near(frames)
        mask = cover(frames)

        for lane in spec.lanes[:2]:
            with self.subTest(lane=lane.name):
                support = lane_support(lane, centre, runs)
                rails = {
                    BlockPos(
                        point.position.x + offset.x,
                        point.position.y + point.y_offset + offset.y,
                        point.position.z + offset.z,
                    )
                    for point in mark_lane(
                        lane, centre, sections["trackbed"], level, support
                    )
                    for offset in (
                        laid_offset(lane.offset, point.laid_at, point.mirrored),
                    )
                    if mask.contains(point.position.x + offset.x,
                                     point.position.z + offset.z)
                }
                self.assertTrue(rails)
                self.assertFalse(rails - support)

                remaining = set(rails)
                reached = {remaining.pop()}
                frontier = list(reached)
                while frontier:
                    here = frontier.pop()
                    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        for dy in (-1, 0, 1):
                            neighbor = BlockPos(
                                here.x + dx, here.y + dy, here.z + dz
                            )
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                reached.add(neighbor)
                                frontier.append(neighbor)
                self.assertEqual(reached, rails)


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


class SectionTestCase(unittest.TestCase):
    def sectioned(self, spans=None, **overrides) -> dict:
        data = config(
            path={
                "points": [[n, 64, 0] for n in range(5)],
                "gradients": [[1, 0, 0]] * 5,
            },
            marker={
                "start": [0, 0],
                "bridge_in": [1, 0],
                "bridge_out": [2, 0],
                "brim_in": [3, 0],
                "end": [4, 0],
            },
        )
        data["section"] = [
            {
                "name": "standard",
                "domain": "rail",
                "schematic": "schems/stdtrack.schem",
                "width": 21,
                "depth": 7,
                **overrides,
            }
        ]
        data["assignment"] = [
            {"from": start, "to": stop, "sections": ["standard"]}
            for start, stop in ([["start", "bridge_in"]] if spans is None else spans)
        ]
        return data



class SectionTests(SectionTestCase):
    def assigned(self, **overrides) -> dict:
        data = self.sectioned()
        data["assignment"] = [{"sections": ["standard"], **overrides}]
        return data

    def test_a_section_says_what_it_is_not_where_it_goes(self) -> None:
        section = parse(self.sectioned()).section("standard")

        self.assertEqual(section.schematic, Path("schems/stdtrack.schem"))
        self.assertEqual(section.domain, "rail")
        self.assertEqual(section.width, 21)
        self.assertEqual(section.elevation, SectionElevation.FOLLOW)

    def test_an_assignment_says_where_a_section_goes(self) -> None:
        assignment = parse(self.sectioned()).assignments[0]

        self.assertEqual(assignment.sections, ("standard",))
        self.assertEqual(str(assignment.span), "start..bridge_in")

    def test_terraform_span_uses_every_assigned_run_of_a_section(self) -> None:
        data = self.sectioned(spans=[
            ["start", "bridge_in"], ["brim_in", "end"],
        ])
        data["domain"].append({"name": "terrain"})
        with TemporaryDirectory() as directory:
            (Path(directory) / "terraform.toml").write_text(
                '[[span]]\nsection = "standard"\nprofile = "embankment"\n'
            )
            data["terraform"] = "terraform.toml"
            spec = parse(data, Path(directory))
        path = build_path(spec)
        assigned = resolve_spans(spec, path)

        stretches = spec.terraform.stretches(
            lambda where: where, len(path.path_points), assigned
        )
        self.assertEqual(
            [(stretch.start, stretch.stop) for stretch in stretches],
            list(assigned["standard"]),
        )
        self.assertEqual(len(stretches), 2)

    def test_a_stretch_may_give_its_own_columns(self) -> None:
        spec = parse(self.assigned(**{"from": [0, 0], "to": [2, 0]}))

        self.assertEqual(str(spec.assignments[0].span), "(0, 0)..(2, 0)")

    def test_one_stretch_may_carry_several_sections(self) -> None:
        data = self.sectioned()
        data["section"].append(
            {"name": "brim", "domain": "rail", "schematic": "brim.schem"}
        )
        data["assignment"][0]["sections"].append("brim")

        self.assertEqual(
            parse(data).assignments[0].sections, ("standard", "brim")
        )

    def test_elevation_is_read_by_name(self) -> None:
        section = parse(self.sectioned(elevation="min")).section("standard")

        self.assertEqual(section.elevation, SectionElevation.MIN)

    def test_an_unknown_marker_names_what_wanted_it(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "a stretch assigned standard names marker 'nowhere'"
        ):
            parse(self.assigned(**{"from": "start", "to": "nowhere"}))

    def test_a_stretch_needs_a_start_and_a_stop(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required key 'to'"):
            parse(self.assigned(**{"from": "start"}))

    def test_a_stretch_may_not_end_where_it_starts(self) -> None:
        with self.assertRaisesRegex(ValueError, "ends where it starts"):
            parse(self.assigned(**{"from": "start", "to": "start"}))

    def test_a_stretch_must_be_assigned_something(self) -> None:
        data = self.sectioned()
        data["assignment"][0]["sections"] = []

        with self.assertRaisesRegex(ValueError, "assigned no sections"):
            parse(data)

    def test_an_assignment_must_name_a_section_that_exists(self) -> None:
        data = self.sectioned()
        data["assignment"][0]["sections"] = ["nowhere"]

        with self.assertRaisesRegex(KeyError, "No section named 'nowhere'"):
            parse(data)




class SpanResolutionTests(SectionTestCase):
    def test_shared_marker_belongs_to_incoming_section(self) -> None:
        data = self.sectioned()
        data["section"].append({**data["section"][0], "name": "incoming"})
        data["assignment"] = [
            {"from": "start", "to": "bridge_out", "sections": ["standard"]},
            {"from": "bridge_out", "to": "end", "sections": ["incoming"]},
        ]
        spec = parse(data)

        self.assertEqual(
            resolve_spans(spec, build_path(spec)),
            {"standard": ((0, 1),), "incoming": ((2, 4),)},
        )

    def resolve(self, spans=None, **overrides):
        data = self.sectioned(spans, **overrides)
        spec = parse(data)
        return resolve_spans(spec, build_path(spec))

    def test_spans_become_path_indices(self) -> None:
        self.assertEqual(self.resolve(), {"standard": ((0, 1),)})

    def test_disjoint_spans_are_kept_in_path_order(self) -> None:
        resolved = self.resolve(
            spans=[["brim_in", "end"], ["start", "bridge_in"]]
        )

        self.assertEqual(resolved, {"standard": ((0, 1), (3, 4))})

    def test_overlapping_stretches_of_one_section_become_one(self) -> None:
        resolved = self.resolve(
            spans=[["start", "bridge_out"], ["bridge_in", "end"]]
        )

        self.assertEqual(resolved, {"standard": ((0, 4),)})

    def test_stretches_that_meet_become_one_sweep(self) -> None:
        # Spans are inclusive at both ends, so a section carried along two
        # stretches in a row is one sweep, not two that collide at the join.
        resolved = self.resolve(
            spans=[["start", "bridge_in"], ["bridge_in", "bridge_out"]]
        )

        self.assertEqual(resolved, {"standard": ((0, 2),)})

    def test_a_span_may_not_run_backwards(self) -> None:
        with self.assertRaisesRegex(ValueError, "runs backwards"):
            self.resolve(spans=[["bridge_out", "start"]])

    def test_a_marker_off_the_path_is_refused(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"but \(9, 9\) is not a point on the path"
        ):
            self.resolve(spans=[["start", [9, 9]]])

    def test_sections_may_stack_on_the_same_stretch(self) -> None:
        data = self.sectioned()
        data["section"].append(
            {
                "name": "standardbrim",
                "domain": "rail",
                "schematic": "schems/stdbrim1.schem",
            }
        )
        data["assignment"][0]["sections"].append("standardbrim")
        spec = parse(data)

        resolved = resolve_spans(spec, build_path(spec))

        self.assertEqual(resolved, {"standard": ((0, 1),), "standardbrim": ((0, 1),)})



def profile_schematic(directory: Path, name="stdtrack", blocks=None,
                      block="minecraft:stone") -> Path:
    # A 3-wide, 1-deep profile, written as a real .schem. The Offset is
    # what centres it on the path, so it is written too.
    registry = BlockRegistry()
    field = BlockField(registry.palette)
    for position in blocks or [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]:
        field.set_block(position, BlockState.create(block))
    directory.mkdir(parents=True, exist_ok=True)
    region = BlockRegion.from_field(field, name=name)
    return write_schematic(
        region, directory / f"{name}.schem",
        data_version=3700, offset=region.origin,
    )


class TerraformSectionSpanBuildTests(SectionTestCase):
    profile = staticmethod(profile_schematic)

    def test_build_resolves_terraform_section_spans(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.sectioned(
                width=3, depth=1,
                spans=[["start", "bridge_in"], ["brim_in", "end"]],
            )
            data["domain"].append({"name": "terrain"})
            data["path"]["lead"] = 2
            data["terraform"] = "terraform.toml"
            (Path(directory) / "terraform.toml").write_text(
                '[[span]]\nsection = "standard"\nprofile = "embankment"\n'
            )
            self.profile(Path(directory) / "schems")
            spec = parse(data, Path(directory))
            expected = resolve_spans(spec, build_path(spec))["standard"]
            built = build_assembly(spec)

        self.assertIsNotNone(built.terraforming)
        self.assertEqual(
            [(start, stop) for _, start, stop, _ in built.terraforming.report.stretches],
            list(expected),
        )


class SchematicTests(SectionTestCase):
    profile = staticmethod(profile_schematic)

    def test_overlapping_section_does_not_split_stair_from_support(self) -> None:
        with TemporaryDirectory() as directory:
            schems = Path(directory) / "schems"
            schems.mkdir()
            data = self.sectioned(width=3, depth=1)
            data["path"]["lead"] = 2
            data["section"][0]["schematic"] = "schems/stairs.schem"
            data["section"].append({
                "name": "brim", "domain": "rail",
                "schematic": "schems/brim.schem", "width": 1, "depth": 1,
            })
            data["assignment"][0]["sections"].append("brim")
            palette = BlockRegistry().palette
            stairs = BlockField(palette)
            for x in (-1, 0, 1):
                stairs.set_block((x, 0, 0), BlockState.create("minecraft:smooth_stone"))
            stairs.set_block(
                (1, 1, 0), BlockState.create(
                    "minecraft:stone_brick_stairs",
                    {"facing": "east", "half": "bottom", "shape": "straight"},
                ),
            )
            brim = BlockField(palette)
            brim.set_block((1, 0, 0), BlockState.create("minecraft:grass_block"))
            for name, field in (("stairs", stairs), ("brim", brim)):
                region = BlockRegion.from_field(field, name=name)
                write_schematic(
                    region, schems / f"{name}.schem",
                    data_version=3700, offset=region.origin,
                )

            world = assemble(parse(data, Path(directory)))
            stair_positions = [
                position for position in world.iter_source_positions()
                if world.registry.get_state(world.get_block_id(position)).name
                == "minecraft:stone_brick_stairs"
            ]
            self.assertTrue(stair_positions)
            for position in stair_positions:
                support = world.registry.get_state(
                    world.get_block_id(position + BlockPos(0, -1, 0))
                )
                self.assertEqual(support.name, "minecraft:smooth_stone")

    def test_edge_columns_keep_their_full_vertical_stacks(self) -> None:
        for edge_from in (None, 1):
            with self.subTest(edge_from=edge_from), TemporaryDirectory() as directory:
                data = self.sectioned(width=3, depth=1)
                data["domain"].extend([
                    {"name": "left_edge"}, {"name": "right_edge"},
                ])
                data["section"][0].update(
                    schematic="schems/edges.schem",
                    edge_domains=["left_edge", "right_edge"],
                )
                if edge_from is not None:
                    data["section"][0]["edge_from"] = edge_from
                self.profile(
                    Path(directory) / "schems", "edges",
                    blocks=[
                        (-1, 0, 0), (-1, 1, 0), (0, 0, 0),
                        (1, 0, 0), (1, 1, 0),
                    ],
                )
                spec = parse(data, Path(directory))
                section = build_sections(
                    spec, BlockRegistry(), build_domains(spec),
                    resolve_spans(spec, build_path(spec)),
                )["standard"]

                edge = [point for point in section.points if point.clip_to_span]
                self.assertEqual(
                    {point.feature.domain.name for point in edge},
                    {"left_edge", "right_edge"},
                )
                self.assertTrue(all(
                    {local.y for local, _ in point.feature.blocks.iter_block_ids()}
                    == {0, 1}
                    for point in edge
                ))
                core = [point for point in section.points if not point.clip_to_span]
                self.assertEqual([point.name for point in core], ["standard"])

    def built(self, directory, **overrides):
        data = self.sectioned(**overrides)
        data["section"][0]["schematic"] = "schems/stdtrack.schem"
        self.profile(Path(directory) / "schems")
        spec = parse(data, Path(directory))
        registry = BlockRegistry()
        domains = build_domains(spec)
        return spec, registry, domains, build_path(spec)

    def test_a_schematic_becomes_the_sections_swept_profile(self) -> None:
        with TemporaryDirectory() as directory:
            spec, registry, domains, path = self.built(directory, width=3, depth=1)

            section = build_sections(
                spec, registry, domains, resolve_spans(spec, path)
            )["standard"]

        # A point per column of the profile, so a diagonal has halves to
        # stagger. The column on the centreline is named for the section.
        self.assertEqual(len(section.points), 3)
        self.assertEqual(
            sorted(each.name for each in section.points),
            ["standard", "standard[-1,0]", "standard[1,0]"],
        )
        self.assertEqual(
            {each.section for each in section.points}, {"standard"}
        )
        self.assertEqual(
            [len(each.feature.blocks) for each in section.points], [1, 1, 1]
        )

    def test_columns_step_the_way_the_section_says(self) -> None:
        # A solid row is chebyshev by default: its columns already share
        # faces along the row, and stepping would push the ones beyond them.
        with TemporaryDirectory() as directory:
            spec, registry, domains, path = self.built(directory, width=3, depth=1)

            section = build_sections(
                spec, registry, domains, resolve_spans(spec, path)
            )["standard"]

        self.assertEqual(
            {each.step_mode for each in section.points}, {StepMode.CHEBYSHEV}
        )

    def test_a_column_may_be_told_to_step(self) -> None:
        # A column that has to stay face-connected along the path says so.
        with TemporaryDirectory() as directory:
            spec, registry, domains, path = self.built(
                directory, width=3, depth=1, steps={"1,0": "manhattan"}
            )

            section = build_sections(
                spec, registry, domains, resolve_spans(spec, path)
            )["standard"]

        modes = {tuple(each.position): each.step_mode for each in section.points}
        self.assertEqual(modes[(1, 0, 0)], StepMode.MANHATTAN)
        self.assertEqual(modes[(0, 0, 0)], StepMode.CHEBYSHEV)

    def test_a_badly_written_column_says_how_to_write_it(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.sectioned(width=3, depth=1, steps={"middle": "manhattan"})
            self.profile(Path(directory) / "schems")

            with self.assertRaisesRegex(ValueError, 'a column is written "x,z"'):
                parse(data, Path(directory))

    def test_a_section_takes_its_domain(self) -> None:
        with TemporaryDirectory() as directory:
            spec, registry, domains, path = self.built(directory, width=3, depth=1)

            section = build_sections(
                spec, registry, domains, resolve_spans(spec, path)
            )["standard"]

        annotation = next(iter(section.points))
        self.assertEqual(annotation.feature.domain, domains.get("rail"))

    def test_a_declared_width_must_match_the_schematic(self) -> None:
        with TemporaryDirectory() as directory:
            spec, registry, domains, path = self.built(directory, width=21)

            with self.assertRaisesRegex(
                ValueError,
                "declares a width of 21, but its schematic is 3 across",
            ):
                build_sections(spec, registry, domains, resolve_spans(spec, path))

    def test_an_undeclared_extent_is_not_checked(self) -> None:
        with TemporaryDirectory() as directory:
            spec, registry, domains, path = self.built(directory, width=0, depth=0)

            self.assertIn(
                "standard",
                build_sections(spec, registry, domains, resolve_spans(spec, path)),
            )

    def test_a_missing_schematic_says_where_it_looked(self) -> None:
        data = self.sectioned()
        spec = parse(data, Path("/nowhere"))

        with self.assertRaisesRegex(
            ValueError, "section 'standard' has no schematic at /nowhere"
        ):
            build_sections(
                spec, BlockRegistry(), build_domains(spec),
                resolve_spans(spec, build_path(spec)),
            )

    def test_spans_become_a_schedule_of_the_section(self) -> None:
        with TemporaryDirectory() as directory:
            spec, registry, domains, path = self.built(directory, width=3, depth=1)

            schedules = build_schedules(
            build_sections(spec, registry, domains, resolve_spans(spec, path)),
            resolve_spans(spec, path),
            len(path.path_points),
        )

        schedule = schedules["standard"]
        self.assertEqual(
            [(span.path_start, span.path_stop) for span in schedule.spans], [(0, 1)]
        )
        self.assertIsNotNone(schedule.at(0))
        self.assertIsNone(schedule.at(3))

    def test_stacked_sections_each_get_their_own_schedule(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.sectioned(width=3, depth=1)
            data["section"].append(
                {
                    "name": "standardbrim",
                    "domain": "rail",
                    "schematic": "schems/stdbrim1.schem",
                }
            )
            data["assignment"][0]["sections"].append("standardbrim")
            self.profile(Path(directory) / "schems")
            self.profile(Path(directory) / "schems", "stdbrim1")
            spec = parse(data, Path(directory))
            registry = BlockRegistry()
            domains = build_domains(spec)
            path = build_path(spec)
            spans = resolve_spans(spec, path)

            sections = build_sections(spec, registry, domains, spans)
            schedules = build_schedules(sections, spans, len(path.path_points))

        self.assertEqual(sorted(schedules), ["standard", "standardbrim"])



class FrameTests(SchematicTests):
    def framed(self, directory, points=None, **overrides):
        # A single-block profile, so a lead of one point is enough for it.
        data = self.sectioned(width=1, depth=1, **overrides)
        data["path"]["lead"] = 1
        if points is not None:
            data["path"]["points"] = points
        self.profile(Path(directory) / "schems", blocks=[(0, 0, 0)])
        spec = parse(data, Path(directory))
        registry = BlockRegistry()
        domains = build_domains(spec)
        path = build_path(spec)
        sections = build_sections(spec, registry, domains, resolve_spans(spec, path))
        schedules = build_schedules(
            sections, resolve_spans(spec, path), len(path.path_points)
        )
        return spec, schedules, path

    def test_a_level_path_is_one_elevation_frame(self) -> None:
        with TemporaryDirectory() as directory:
            spec, schedules, path = self.framed(directory)
            sections = [s.section for q in schedules.values() for s in q.spans]

            elevation = build_elevation_frames(spec, path, sections)

        self.assertEqual(len(elevation), 1)
        self.assertEqual(elevation[0].y_offset, 64)

    def test_the_path_changing_height_breaks_it_into_frames(self) -> None:
        with TemporaryDirectory() as directory:
            spec, schedules, path = self.framed(
                directory,
                points=[[0, 64, 0], [1, 64, 0], [2, 66, 0], [3, 66, 0], [4, 66, 0]],
            )
            sections = [s.section for q in schedules.values() for s in q.spans]

            elevation = build_elevation_frames(spec, path, sections)

        self.assertEqual([frame.y_offset for frame in elevation], [64, 66])
        # Index 3, not 2: a lead of one point shifts the path proper along.
        self.assertEqual(elevation[1].path_start, 3)

    def test_section_frames_cut_at_levels_and_at_span_ends(self) -> None:
        with TemporaryDirectory() as directory:
            spec, schedules, path = self.framed(
                directory,
                points=[[0, 64, 0], [1, 64, 0], [2, 64, 0], [3, 66, 0], [4, 66, 0]],
                spans=[["start", "bridge_in"]],
            )

            frames = build_path_frames(spec, schedules, path)

        # Cut where the section runs out (3) and where the level changes (4).
        self.assertEqual([frame.path_start for frame in frames], [1, 3, 4])
        self.assertEqual([frame.y_offset for frame in frames], [64, 64, 66])

    def test_every_frame_clips_its_own_columns(self) -> None:
        with TemporaryDirectory() as directory:
            spec, schedules, path = self.framed(directory)

            frames = build_path_frames(spec, schedules, path)

        # A frame is a ColumnMask: adjacent frames share a boundary and the
        # half-open convention puts each column in exactly one of them.
        inside = [frame for frame in frames if frame.contains(0.0, 0.0)]
        self.assertEqual(len(inside), 1)

    def test_a_short_lead_says_how_much_is_needed(self) -> None:
        with TemporaryDirectory() as directory:
            # A five-wide profile reaches further than a lead of nothing.
            data = self.sectioned(width=5, depth=1)
            self.profile(
                Path(directory) / "schems",
                blocks=[(x, 0, 0) for x in range(-2, 3)],
            )
            spec = parse(data, Path(directory))
            registry = BlockRegistry()
            domains = build_domains(spec)
            path = build_path(spec)
            sections = build_sections(spec, registry, domains, resolve_spans(spec, path))
            schedules = build_schedules(
            sections, resolve_spans(spec, path), len(path.path_points)
        )

            with self.assertRaisesRegex(
                ValueError, r"need a lead of \d+ points at each end, and the path gives 0"
            ):
                build_path_frames(spec, schedules, path)



class SweepTests(SchematicTests):
    def swept(self, directory, **overrides):
        data = self.sectioned(width=3, depth=1, **overrides)
        data["path"]["points"] = [[n, 64, 0] for n in range(9)]
        data["path"]["gradients"] = [[1, 0, 0]] * 9
        data["path"]["lead"] = 3
        self.profile(Path(directory) / "schems")
        spec = parse(data, Path(directory))
        registry = BlockRegistry()
        domains = build_domains(spec)
        path = build_path(spec)
        sections = build_sections(spec, registry, domains, resolve_spans(spec, path))
        schedules = build_schedules(
            sections, resolve_spans(spec, path), len(path.path_points)
        )
        frames = build_path_frames(spec, schedules, path)
        return sweep(spec, schedules, path, frames)

    def test_a_section_sweeps_into_one_run_per_span(self) -> None:
        with TemporaryDirectory() as directory:
            runs = self.swept(directory, spans=[["start", "bridge_in"],
                                                ["brim_in", "end"]])

        # One run per column per span, and every one of them the same section.
        self.assertEqual(sorted({run.span for run in runs}), [0, 1])
        self.assertEqual({run.annotation.section for run in runs}, {"standard"})
        self.assertEqual(len(runs), 6)

    def test_a_run_keeps_the_annotations_own_feature(self) -> None:
        with TemporaryDirectory() as directory:
            runs = self.swept(directory)

        # Each run carries its own column's blocks, and the three columns
        # together are the profile.
        self.assertEqual(
            {run.annotation.feature.elevation_mode for run in runs},
            {ElevationMode.RIGID},
        )
        self.assertEqual(
            sum(len(run.annotation.feature.blocks) for run in runs), 3
        )

    def test_every_delivered_point_carries_its_frames_level(self) -> None:
        with TemporaryDirectory() as directory:
            runs = self.swept(
                directory,
                spans=[["start", "end"]],
            )

        levels = {point.y_offset for run in runs for point in run.path}
        self.assertEqual(levels, {64})

    def test_a_climb_gives_a_run_points_at_both_levels(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.sectioned(width=3, depth=1, spans=[["start", "end"]])
            data["path"]["points"] = [[n, 64 + (n >= 3), 0] for n in range(9)]
            data["path"]["gradients"] = [[1, 0, 0]] * 9
            data["path"]["lead"] = 3
            self.profile(Path(directory) / "schems")
            spec = parse(data, Path(directory))
            registry = BlockRegistry()
            domains = build_domains(spec)
            path = build_path(spec)
            sections = build_sections(spec, registry, domains, resolve_spans(spec, path))
            schedules = build_schedules(
            sections, resolve_spans(spec, path), len(path.path_points)
        )
            frames = build_path_frames(spec, schedules, path)

            runs = sweep(spec, schedules, path, frames)

        # A run per column, each carrying both levels across the break.
        self.assertEqual(len(runs), 3)
        for run in runs:
            self.assertEqual(
                {point.y_offset for point in run.path}, {64, 65},
                f"{run.annotation.name} does not cross the break",
            )

    def test_stacked_sections_sweep_into_separate_runs(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.sectioned(width=3, depth=1)
            data["path"]["points"] = [[n, 64, 0] for n in range(9)]
            data["path"]["gradients"] = [[1, 0, 0]] * 9
            data["path"]["lead"] = 3
            data["section"].append(
                {
                    "name": "standardbrim",
                    "domain": "rail",
                    "width": 3,
                    "depth": 1,
                    "schematic": "schems/stdbrim1.schem",
                }
            )
            data["assignment"][0]["sections"].append("standardbrim")
            self.profile(Path(directory) / "schems")
            self.profile(Path(directory) / "schems", "stdbrim1")
            spec = parse(data, Path(directory))
            registry = BlockRegistry()
            domains = build_domains(spec)
            path = build_path(spec)
            sections = build_sections(spec, registry, domains, resolve_spans(spec, path))
            schedules = build_schedules(
            sections, resolve_spans(spec, path), len(path.path_points)
        )
            frames = build_path_frames(spec, schedules, path)

            runs = sweep(spec, schedules, path, frames)

        self.assertEqual(
            sorted({run.annotation.section for run in runs}),
            ["standard", "standardbrim"],
        )

    def test_a_swept_section_reaches_the_world(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.sectioned(width=3, depth=1, spans=[["start", "end"]])
            data["path"]["points"] = [[n, 64, 0] for n in range(9)]
            data["path"]["gradients"] = [[1, 0, 0]] * 9
            data["path"]["lead"] = 3
            self.profile(Path(directory) / "schems")

            world = assemble(parse(data, Path(directory)))

        # Three columns wide, across travel rather than along it.
        columns = {(position.x, position.z) for position in world.iter_source_positions()}
        self.assertEqual({z for _, z in columns}, {-1, 0, 1})



class StructureTests(SweepTests):
    def structured(self, directory, **overrides):
        data = self.sectioned(width=3, depth=1, spans=[["start", "end"]])
        data["path"]["points"] = [[n, 64, 0] for n in range(9)]
        data["path"]["gradients"] = [[1, 0, 0]] * 9
        data["path"]["lead"] = 3
        data["marker"]["end"] = [8, 0]
        data["structure"] = [
            {
                "name": "pillar",
                "domain": "rail",
                "sections": ["standard"],
                "headings": {"0.0": "schems/pillar.schem"},
                **overrides,
            }
        ]
        self.profile(Path(directory) / "schems")
        self.profile(Path(directory) / "schems", "pillar", blocks=[(0, 0, 0)],
                     block="minecraft:gold_block")
        return data

    def test_a_structure_names_its_sections_and_schematics(self) -> None:
        with TemporaryDirectory() as directory:
            spec = parse(self.structured(directory), Path(directory))

        structure = spec.structure("pillar")
        self.assertEqual(structure.sections, ("standard",))
        self.assertEqual(structure.resolution, 90.0)
        self.assertEqual(list(structure.headings), [0.0])

    def test_one_schematic_covers_every_quadrant(self) -> None:
        # A resolution of 90 has one authored heading; quarter turns do the rest.
        with TemporaryDirectory() as directory:
            spec = parse(self.structured(directory), Path(directory))
            domains = build_domains(spec)

            stamp = build_structure(spec.structure("pillar"), BlockRegistry(), domains)

        self.assertEqual(stamp.headings, (0.0,))
        self.assertEqual(stamp.at(180.0)[1], 2)

    def test_an_unquoted_heading_says_to_quote_it(self) -> None:
        # TOML reads a bare 0.0 as a dotted key, so it arrives nested.
        with TemporaryDirectory() as directory:
            data = self.structured(directory)
            data["structure"][0]["headings"] = {"0": {"0": "schems/pillar.schem"}}

            with self.assertRaisesRegex(ValueError, "write it quoted"):
                parse(data, Path(directory))

    def test_a_finer_resolution_needs_every_heading(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.structured(directory, resolution=22.5)

            with self.assertRaisesRegex(
                ValueError, r"missing \[22.5, 45.0\]"
            ):
                parse(data, Path(directory))

    def test_a_structure_must_ride_on_something(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.structured(directory, sections=[])

            with self.assertRaisesRegex(ValueError, "assigned to no sections"):
                parse(data, Path(directory))

    def test_an_unknown_section_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.structured(directory, sections=["nowhere"])

            with self.assertRaisesRegex(KeyError, "No section named 'nowhere'"):
                parse(data, Path(directory))

    def test_a_negative_spacing_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.structured(directory, spacing=-1)

            with self.assertRaisesRegex(ValueError, "negative spacing"):
                parse(data, Path(directory))

    def test_an_angle_must_be_a_turn(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.structured(directory, angle=200)

            with self.assertRaisesRegex(ValueError, "between 0 and 180 degrees"):
                parse(data, Path(directory))

    def test_a_bend_earns_a_structure_of_its_own(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.structured(directory, spacing=0, angle=45)
            # Straight for five, then a diagonal.
            data["path"]["points"] = [
                *[[n, 64, 0] for n in range(5)],
                *[[5 + n, 64, -n] for n in range(1, 4)],
            ]
            data["path"]["gradients"] = [*[[1, 0, 0]] * 5, *[[1, 0, -1]] * 3]
            data["marker"]["end"] = [7, -2]

            world = assemble(parse(data, Path(directory)))
            marks = sorted(
                (position.x, position.z)
                for position in world.iter_source_positions()
                if world.registry.get_state(
                    world.get_block_id(position)
                ).name.endswith("gold_block")
            )

        # One where the run starts, one where it turns, one where it ends.
        self.assertEqual(len(marks), 3)
        self.assertEqual(marks[0], (0, 0))

    def test_it_rides_only_the_runs_of_its_own_sections(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.structured(directory)
            spec = parse(data, Path(directory))
            registry = BlockRegistry()
            domains = build_domains(spec)
            path = build_path(spec)
            sections = build_sections(spec, registry, domains, resolve_spans(spec, path))
            schedules = build_schedules(
            sections, resolve_spans(spec, path), len(path.path_points)
        )
            frames = build_path_frames(spec, schedules, path)
            runs = sweep(spec, schedules, path, frames)

        carried = runs_under(spec, spec.structure("pillar"), runs)
        self.assertEqual([run.annotation.name for run in carried], ["standard"])
        self.assertEqual(runs_under(spec, spec.structure("pillar"), ()), ())

    def test_spacing_thins_a_structure_along_its_section(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(parse(self.structured(directory, spacing=3),
                                   Path(directory)))
            marks = sorted(
                position.x
                for position in world.iter_source_positions()
                if world.registry.get_state(
                    world.get_block_id(position)
                ).name.endswith("gold_block")
            )

        # Every three blocks along, rather than at every point, and the end.
        self.assertEqual(marks, [0, 3, 6, 8])

    def test_structure_spacing_continues_across_section_swaps(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.structured(directory, spacing=3)
            data["marker"]["mid"] = [4, 0]
            data["section"].append({**data["section"][0], "name": "other"})
            data["assignment"] = [
                {"from": "start", "to": "mid", "sections": ["standard"]},
                {"from": "mid", "to": "end", "sections": ["other"]},
            ]
            data["structure"][0]["sections"] = ["standard", "other"]

            world = assemble(parse(data, Path(directory)))
            marks = sorted(
                position.x
                for position in world.iter_source_positions()
                if world.registry.get_state(world.get_block_id(position)).name
                == "minecraft:gold_block"
            )

        self.assertEqual(marks, [0, 3, 6, 8])

    def test_without_spacing_it_lands_at_every_point(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(parse(self.structured(directory), Path(directory)))
            marks = sorted(
                position.x
                for position in world.iter_source_positions()
                if world.registry.get_state(
                    world.get_block_id(position)
                ).name.endswith("gold_block")
            )

        self.assertEqual(marks, [0, 1, 2, 3, 4, 5, 6, 7, 8])

    def test_an_offset_turns_with_the_structure(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(parse(self.structured(directory, spacing=8,
                                                   offset=[2, 0, 0]),
                                   Path(directory)))
            marks = [
                tuple(position)
                for position in world.iter_source_positions()
                if world.registry.get_state(
                    world.get_block_id(position)
                ).name.endswith("gold_block")
            ]

        # Travelling east, two to the right of travel is two south. The run
        # ends anchored, so the far end carries one too.
        self.assertEqual(marks, [(0, 64, 2), (8, 64, 2)])



def pieces(cells, faces: bool) -> int:
    # Connected pieces, joined by faces or by any touch.
    near = [
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (abs(dx) + abs(dy) + abs(dz) == 1 if faces else (dx, dy, dz) != (0, 0, 0))
    ]
    left, count = set(cells), 0
    while left:
        stack = [left.pop()]
        count += 1
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in near:
                if (x + dx, y + dy, z + dz) in left:
                    left.remove((x + dx, y + dy, z + dz))
                    stack.append((x + dx, y + dy, z + dz))
    return count


class WireTests(SweepTests):
    def wired(self, directory, **overrides):
        data = self.sectioned(width=3, depth=1, spans=[["start", "end"]])
        data["path"]["points"] = [[n, 64, 0] for n in range(9)]
        data["path"]["gradients"] = [[1, 0, 0]] * 9
        data["path"]["lead"] = 3
        data["marker"]["end"] = [8, 0]
        data["wire"] = [
            {
                "name": "catenary",
                "domain": "rail",
                "sections": ["standard"],
                "schematic": "schems/wire.schem",
                "geometry": "line",
                "spacing": 4,
                **overrides,
            }
        ]
        self.profile(Path(directory) / "schems")
        self.profile(Path(directory) / "schems", "wire", blocks=[(0, 0, 0)],
                     block="minecraft:tripwire")
        return data

    def prepared(self, directory, **overrides):
        spec = parse(self.wired(directory, **overrides), Path(directory))
        registry = BlockRegistry()
        domains = build_domains(spec)
        path = build_path(spec)
        sections = build_sections(spec, registry, domains, resolve_spans(spec, path))
        schedules = build_schedules(
            sections, resolve_spans(spec, path), len(path.path_points)
        )
        frames = build_path_frames(spec, schedules, path)
        runs = sweep(spec, schedules, path, frames)
        return spec, runs, registry, domains, sections

    def strand(self, world):
        return sorted(
            tuple(position)
            for position in world.iter_source_positions()
            if world.registry.get_state(
                world.get_block_id(position)
            ).name.endswith("tripwire")
        )

    def test_a_wire_names_its_sections_and_stepping(self) -> None:
        with TemporaryDirectory() as directory:
            spec = parse(self.wired(directory), Path(directory))

        wire = spec.wire("catenary")
        self.assertEqual(wire.sections, ("standard",))
        self.assertEqual(wire.geometry, WireGeometry.LINE)
        self.assertEqual(wire.step, StepMode.MANHATTAN)
        self.assertEqual(wire.spacing, 4)

    def test_stepping_is_read_by_name(self) -> None:
        with TemporaryDirectory() as directory:
            spec = parse(self.wired(directory, step="chebyshev"), Path(directory))

        self.assertEqual(spec.wire("catenary").step, StepMode.CHEBYSHEV)

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not one of MANHATTAN, CHEBYSHEV"):
                parse(self.wired(directory, step="manhattan_3d"), Path(directory))

    def test_a_wire_follows_its_column_unless_told_otherwise(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.wired(directory)
            del data["wire"][0]["geometry"], data["wire"][0]["spacing"]
            spec = parse(data, Path(directory))

        self.assertEqual(spec.wire("catenary").geometry, WireGeometry.FOLLOW)

    def test_anchors_ask_for_a_line(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.wired(directory)
            del data["wire"][0]["geometry"]

            with self.assertRaisesRegex(ValueError, 'geometry = "line"'):
                parse(data, Path(directory))

    def test_a_followed_wire_has_no_anchors_to_place(self) -> None:
        for key, value in (("spacing", 4), ("angle", 15), ("measure", "index")):
            with self.subTest(key=key), TemporaryDirectory() as directory:
                data = self.wired(directory, geometry="follow")
                del data["wire"][0]["spacing"]
                data["wire"][0][key] = value

                with self.assertRaisesRegex(
                    ValueError, "only place the anchors of a line"
                ):
                    parse(data, Path(directory))

    def test_a_wire_must_ride_on_something(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not both and not neither"):
                parse(self.wired(directory, sections=[]), Path(directory))

    def test_a_negative_spacing_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "wire 'catenary' has a negative"):
                parse(self.wired(directory, spacing=-1), Path(directory))

    def test_a_wire_runs_from_pole_to_pole(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(parse(self.wired(directory), Path(directory)))

        # Poles every four blocks along a run of nine: an unbroken strand
        # from the first to the last, not a block at each pole.
        self.assertEqual(
            self.strand(world),
            [(x, 64, 0) for x in range(0, 9)],
        )

    def test_an_offset_hangs_the_wire_off_the_path(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(
                parse(self.wired(directory, offset=[1, 3, 0]), Path(directory))
            )

        # Three up and one to the right of travel, which heading east is south.
        self.assertEqual(self.strand(world), [(x, 67, 1) for x in range(0, 9)])

    def test_a_wire_is_one_placement_however_long_it_runs(self) -> None:
        with TemporaryDirectory() as directory:
            spec, runs, registry, domains, sections = self.prepared(directory)
            world = assemble(spec)

        wires = [
            placement
            for placement in world.instances
            if placement.connectivity_domain == domains.id_of("rail")
            and len(placement.field) > 3
        ]
        self.assertEqual(len(wires), 1)

    def test_a_wire_spans_its_run_however_wide_the_spacing(self) -> None:
        # The run's last point is a pole whatever the spacing, so a wire too
        # long to be spaced still reaches from one end to the other.
        with TemporaryDirectory() as directory:
            spec, runs, registry, domains, sections = self.prepared(directory, spacing=100)
            strand = strand_of(
                spec, spec.wire("catenary"), runs[0], sections["standard"],
                lambda x, z, index: 64,
            )

        self.assertEqual(min(strand).x, 0)
        self.assertEqual(max(strand).x, 8)

    def test_manhattan_stepping_keeps_the_strand_face_connected(self) -> None:
        with TemporaryDirectory() as directory:
            spec, runs, registry, domains, sections = self.prepared(
                directory, offset=[0, 2, 0], spacing=4
            )
            # Segments share the pole they meet at, so the strand is deduped.
            blocks = sorted(set(strand_of(
                spec, spec.wire("catenary"), runs[0], sections["standard"],
                lambda x, z, index: 64,
            )))

        for before, after in zip(blocks, blocks[1:]):
            self.assertEqual(
                sum(abs(after[axis] - before[axis]) for axis in range(3)), 1
            )

    def test_a_wire_climbs_between_anchors_independently_of_the_track(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.wired(directory, offset=[0, 3, 0], spacing=100)
            data["path"]["points"] = [
                [x, 64 + (x >= 2), 0] for x in range(9)
            ]
            world = assemble(parse(data, Path(directory)))

        # Manhattan in 3D: the climb takes a block of its own.
        expected = sorted(tuple(point) for point in line(
            (0, 67, 0), (8, 68, 0), LineMode.MANHATTAN
        ))
        self.assertEqual(self.strand(world), expected)
        self.assertIn((5, 67, 0), expected)
        self.assertIn((5, 68, 0), expected)

    def test_a_chebyshev_wire_climbs_across_a_corner(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.wired(
                directory, offset=[0, 3, 0], spacing=100, step="chebyshev",
            )
            data["path"]["points"] = [
                [x, 64 + (x >= 2), 0] for x in range(9)
            ]
            world = assemble(parse(data, Path(directory)))

        self.assertEqual(
            self.strand(world),
            [(x, 67 + (x >= 5), 0) for x in range(9)],
        )

    def followed(self, directory, points, gradients, **overrides):
        data = self.wired(directory, offset=[0, 3, 0], **overrides)
        del data["wire"][0]["geometry"], data["wire"][0]["spacing"]
        data["path"]["points"] = points
        data["path"]["gradients"] = gradients
        data["marker"]["end"] = points[-1][::2]
        return data

    def test_a_followed_wire_climbs_with_the_track(self) -> None:
        climb = [[x, 64 + (x >= 2), 0] for x in range(9)]
        with TemporaryDirectory() as directory:
            world = assemble(parse(
                self.followed(directory, climb, [[1, 0, 0]] * 9), Path(directory)
            ))

        # Up where the track rises, not spread out between anchors; the climb
        # takes a block of its own.
        self.assertEqual(
            self.strand(world),
            [(0, 67, 0), (1, 67, 0), (2, 67, 0)] + [(x, 68, 0) for x in range(2, 9)],
        )

        with TemporaryDirectory() as directory:
            world = assemble(parse(
                self.followed(directory, climb, [[1, 0, 0]] * 9, step="chebyshev"),
                Path(directory),
            ))

        self.assertEqual(
            self.strand(world),
            [(0, 67, 0), (1, 67, 0)] + [(x, 68, 0) for x in range(2, 9)],
        )

    def test_a_followed_wire_keeps_to_the_bend_a_line_cuts(self) -> None:
        bend = [[n, 64, 0] for n in range(9)] + [[8 + n, 64, -n] for n in range(1, 9)]
        gradients = [[1, 0, 0]] * 9 + [[1, 0, -1]] * 8
        with TemporaryDirectory() as directory:
            data = self.followed(directory, bend, gradients)
            followed = self.strand(assemble(parse(data, Path(directory))))
            data = self.followed(directory, bend, gradients, step="chebyshev")
            diagonal = self.strand(assemble(parse(data, Path(directory))))
            data["wire"][0].update(geometry="line", spacing=100)
            cut = self.strand(assemble(parse(data, Path(directory))))

        near_the_path = lambda x, z: any(
            max(abs(x - px), abs(z - pz)) <= 1 for px, _, pz in bend
        )
        for strand in (followed, diagonal):
            self.assertIn((8, 67, 0), strand)
            self.assertTrue(all(near_the_path(x, z) for x, _, z in strand), strand)
        self.assertNotIn((8, 67, 0), cut)

        # Manhattan joins round the bend by faces; chebyshev only by corners.
        self.assertEqual(pieces(followed, faces=True), 1)
        self.assertEqual(pieces(diagonal, faces=False), 1)
        self.assertGreater(pieces(diagonal, faces=True), 1)

    def test_a_run_of_one_point_still_lays_its_block(self) -> None:
        with TemporaryDirectory() as directory:
            spec, runs, registry, domains, sections = self.prepared(directory)
        single = FeaturePath(rules=())
        single.add_point(runs[0].path[0])

        strand = strand_of(
            spec, spec.wire("catenary"), runs[0]._replace(path=single),
            sections["standard"], lambda x, z, index: 64,
        )

        self.assertEqual(len(strand), 1)



class DropperTests(SweepTests):
    def hung(self, directory, points=None, gradients=None, **overrides):
        data = self.sectioned(width=3, depth=1, spans=[["start", "end"]])
        count = 25 if points is None else len(points)
        data["path"]["points"] = points or [[n, 64, 0] for n in range(count)]
        data["path"]["gradients"] = gradients or [[1, 0, 0]] * count
        data["path"]["lead"] = 3
        data["marker"]["end"] = data["path"]["points"][-1][::2]
        strand = lambda name, offset: {
            "name": name, "domain": "rail", "sections": ["standard"],
            "schematic": "schems/wire.schem", "offset": offset,
            "geometry": "line", "spacing": 8, "step": "chebyshev",
        }
        data["wire"] = [
            strand("messenger", [0, 6, 0]),
            strand("contact", [0, 3, 0]),
            {
                "name": "dropper", "domain": "rail",
                "between": ["messenger", "contact"],
                "schematic": "schems/drop.schem", "spacing": 4,
                "step": "manhattan", **overrides,
            },
        ]
        self.profile(Path(directory) / "schems")
        self.profile(Path(directory) / "schems", "wire", blocks=[(0, 0, 0)],
                     block="minecraft:tripwire")
        self.profile(Path(directory) / "schems", "drop", blocks=[(0, 0, 0)],
                     block="minecraft:chain")
        return data

    def chains(self, world):
        return sorted(
            (position.x, position.y, position.z)
            for position in world.iter_source_positions()
            if world.registry.get_state(
                world.get_block_id(position)
            ).name.endswith("chain")
        )

    def test_a_wire_runs_along_sections_or_between_wires(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.hung(directory, sections=["standard"])

            with self.assertRaisesRegex(ValueError, "not both and not neither"):
                parse(data, Path(directory))

    def test_a_dropper_stretches_between_exactly_two(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.hung(directory, between=["messenger"])

            with self.assertRaisesRegex(ValueError, "and names 1"):
                parse(data, Path(directory))

    def test_a_dropper_is_a_line(self) -> None:
        with TemporaryDirectory() as directory:
            spec = parse(self.hung(directory), Path(directory))

        self.assertEqual(spec.wire("dropper").geometry, WireGeometry.LINE)

        with TemporaryDirectory() as directory:
            data = self.hung(directory, geometry="follow")

            with self.assertRaisesRegex(ValueError, "which only a line can"):
                parse(data, Path(directory))

    def test_a_dropper_may_not_hang_from_another_dropper(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.hung(directory)
            data["wire"].append({
                "name": "second", "domain": "rail",
                "between": ["dropper", "messenger"],
                "schematic": "schems/drop.schem",
            })

            with self.assertRaisesRegex(
                ValueError, "which is itself stretched between other wires"
            ):
                parse(data, Path(directory))

    def test_a_dropper_rides_whatever_the_wires_it_joins_ride(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.hung(directory)
            spec = parse(data, Path(directory))

            self.assertEqual(
                fitted_sections(spec, spec.wire("dropper")), ("standard",)
            )

    def test_droppers_hang_at_their_own_spacing(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(parse(self.hung(directory), Path(directory)))

        columns = sorted({x for x, _, _ in self.chains(world)})
        # Every four, and the point the run ends on whatever the spacing.
        self.assertEqual(columns, [0, 4, 8, 12, 16, 20, 24])

    def test_a_dropper_reaches_from_one_wire_to_the_other(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(parse(self.hung(directory), Path(directory)))

        at_start = sorted(y for x, y, _ in self.chains(world) if x == 0)
        self.assertEqual(at_start, [67, 68, 69, 70])

    def test_a_dropper_joins_the_wires_at_their_interpolated_height(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.hung(directory)
            data["path"]["points"] = [
                [x, 64 + (x >= 2), 0] for x in range(25)
            ]
            world = assemble(parse(data, Path(directory)))

        # The deck has already risen at x=4, but each wire spanning x=0..8
        # reaches its new height later. The dropper follows those wire blocks.
        at_four = sorted(y for x, y, _ in self.chains(world) if x == 4)
        self.assertEqual(at_four, [67, 68, 69, 70])

    def test_a_dropper_hangs_from_where_a_wire_actually_is(self) -> None:
        # Poles every eight over a bend: between them the wire lies on the
        # straight line, so a dropper reaches that rather than the offset.
        with TemporaryDirectory() as directory:
            points = [[n, 64, 0] for n in range(9)]
            points += [[8 + n, 64, -n] for n in range(1, 9)]
            gradients = [[1, 0, 0]] * 9 + [[1, 0, -1]] * 8
            world = assemble(
                parse(self.hung(directory, points, gradients), Path(directory))
            )

        # Every dropper still reaches from one wire to the other. Counting
        # the columns would only say how many there are, and a bend moves
        # that about; what matters is that none of them stops short.
        columns: dict[tuple[int, int], list[int]] = {}
        for x, y, z in self.chains(world):
            columns.setdefault((x, z), []).append(y)

        self.assertTrue(columns)
        for where, heights in sorted(columns.items()):
            self.assertEqual(
                sorted(heights), [67, 68, 69, 70],
                f"the dropper at {where} does not reach both wires",
            )



class LaneTests(SweepTests):
    def laned(self, directory, **overrides):
        data = self.sectioned(width=3, depth=1, spans=[["start", "end"]])
        data["path"]["points"] = [[n, 64, 0] for n in range(17)]
        data["path"]["gradients"] = [[1, 0, 0]] * 17
        data["path"]["lead"] = 3
        data["marker"]["end"] = [16, 0]
        data["lane"] = [
            {
                "name": "centre",
                "domain": "rail",
                "sections": ["standard"],
                "block": "minecraft:white_concrete",
                "on": 3,
                "off": 5,
                **overrides,
            }
        ]
        self.profile(Path(directory) / "schems")
        return data

    def marks(self, world):
        return sorted(
            position.x
            for position in world.iter_source_positions()
            if world.registry.get_state(
                world.get_block_id(position)
            ).name.endswith("white_concrete")
        )

    def test_a_lane_is_a_dash_and_a_gap_over_and_over(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(
                parse(self.laned(directory, step="chebyshev"), Path(directory))
            )

        # Three on, five off, from where the run starts.
        self.assertEqual(self.marks(world), [0, 1, 2, 8, 9, 10, 16])

    def test_a_lane_may_be_counted_in_blocks(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(
                parse(
                    self.laned(directory, step="chebyshev", measure="distance"),
                    Path(directory),
                )
            )

        # The run's points are a block apart, so both measures agree here.
        self.assertEqual(self.marks(world), [0, 1, 2, 8, 9, 10, 16])

    def test_a_stepping_lane_paints_a_block_along_travel_as_well(self) -> None:
        # A lane sits at a local column and answers the same step question a
        # column of the section does. Stepping, it paints where it sits and a
        # block along travel, so a dash of three points is four blocks long
        # and consecutive blocks share a face.
        with TemporaryDirectory() as directory:
            world = assemble(parse(self.laned(directory), Path(directory)))

        self.assertEqual(self.marks(world), [0, 1, 2, 3, 8, 9, 10, 11, 16])

    def test_a_lane_takes_a_block_or_a_schematic(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.laned(directory, schematic="schems/stdtrack.schem")

            with self.assertRaisesRegex(ValueError, "not both and not neither"):
                parse(data, Path(directory))

    def test_a_lane_needs_one_of_them(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.laned(directory)
            del data["lane"][0]["block"]

            with self.assertRaisesRegex(ValueError, "not both and not neither"):
                parse(data, Path(directory))

    def test_a_dash_keeps_at_least_one_block(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least one block"):
                parse(self.laned(directory, on=0), Path(directory))

    def test_a_gap_may_be_nothing_at_all(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(parse(self.laned(directory, off=0), Path(directory)))

        self.assertEqual(self.marks(world), list(range(0, 17)))

    def test_a_negative_gap_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no blocks at all"):
                parse(self.laned(directory, off=-1), Path(directory))

    def test_a_lane_is_offset_the_way_the_path_faces(self) -> None:
        with TemporaryDirectory() as directory:
            world = assemble(
                parse(self.laned(directory, offset=[1, 1, 0]), Path(directory))
            )
            columns = sorted(
                (position.x, position.y, position.z)
                for position in world.iter_source_positions()
                if world.registry.get_state(
                    world.get_block_id(position)
                ).name.endswith("white_concrete")
            )

        # One up and one right of travel, which heading east is south.
        self.assertEqual(columns[0], (0, 65, 1))




class BuildOrderTests(SweepTests):
    def ordered(self, directory, deck_order, paint_order):
        data = self.sectioned(width=1, depth=1, spans=[["start", "end"]])
        data["path"]["points"] = [[n, 64, 0] for n in range(9)]
        data["path"]["gradients"] = [[1, 0, 0]] * 9
        data["path"]["lead"] = 3
        data["marker"]["end"] = [8, 0]
        data["domain"] = [
            {"name": "rail", "order": deck_order,
             "placement": {"mode": "OVERWRITE"}},
            {"name": "paint", "order": paint_order,
             "placement": {"mode": "OVERWRITE"}},
        ]
        data["lane"] = [
            {"name": "line", "domain": "paint", "sections": ["standard"],
             "block": "minecraft:white_concrete", "on": 1, "off": 0},
        ]
        self.profile(Path(directory) / "schems", blocks=[(0, 0, 0)])
        return assemble(parse(data, Path(directory)))

    def blocks_in(self, world):
        return {
            world.registry.get_state(world.get_block_id(position)).name
            for position in world.iter_source_positions()
        }

    def test_a_later_domain_covers_an_earlier_one(self) -> None:
        with TemporaryDirectory() as directory:
            world = self.ordered(directory, deck_order=0, paint_order=1)

        self.assertEqual(self.blocks_in(world), {"minecraft:white_concrete"})

    def test_an_earlier_domain_is_covered_by_what_follows(self) -> None:
        with TemporaryDirectory() as directory:
            world = self.ordered(directory, deck_order=1, paint_order=0)

        self.assertEqual(self.blocks_in(world), {"minecraft:stone"})

    def test_a_section_takes_its_domains_placement_mode(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.sectioned(width=3, depth=1)
            data["domain"] = [{"name": "rail",
                               "placement": {"mode": "KEEP_EXISTING"}}]
            self.profile(Path(directory) / "schems")
            spec = parse(data, Path(directory))
            registry = BlockRegistry()
            domains = build_domains(spec)
            spans = resolve_spans(spec, build_path(spec))

            section = build_sections(spec, registry, domains, spans)["standard"]

        annotation = next(iter(section.points))
        self.assertEqual(
            annotation.feature.placement_mode, PlacementMode.KEEP_EXISTING
        )

    def test_a_section_may_override_its_domain(self) -> None:
        with TemporaryDirectory() as directory:
            data = self.sectioned(width=3, depth=1,
                                  placement_mode="REPLACE_AIR_ONLY")
            data["domain"] = [{"name": "rail",
                               "placement": {"mode": "KEEP_EXISTING"}}]
            self.profile(Path(directory) / "schems")
            spec = parse(data, Path(directory))
            registry = BlockRegistry()
            domains = build_domains(spec)
            spans = resolve_spans(spec, build_path(spec))

            section = build_sections(spec, registry, domains, spans)["standard"]

        annotation = next(iter(section.points))
        self.assertEqual(
            annotation.feature.placement_mode, PlacementMode.REPLACE_AIR_ONLY
        )


class OutputTests(SectionTestCase):
    def written(self, directory, name, mode=None, first=BlockPos(0, 64, 0)):
        # The suffix on the output is what picks the format. The path is long
        # enough, and led far enough, for the profile to reach past its ends.
        data = self.sectioned(width=3, depth=1, spans=[["start", "end"]])
        data["path"]["points"] = [
            [first.x + n, first.y, first.z] for n in range(9)
        ]
        data["path"]["gradients"] = [[1, 0, 0]] * 9
        data["path"]["lead"] = 3
        data["marker"] = {
            name: [first.x + position[0], first.z + position[1]]
            for name, position in data["marker"].items()
        }
        data["marker"]["end"] = [first.x + 8, first.z]
        data["output"] = {"path": f"out/{name}", "name": "curve"}
        if mode is not None:
            data["output"]["mode"] = mode
        profile_schematic(Path(directory) / "schems")
        spec = parse(data, Path(directory))
        return write(assemble(spec), spec)

    def test_a_litematic_suffix_writes_a_litematic(self) -> None:
        with TemporaryDirectory() as directory:
            path = self.written(directory, "curve.litematic")

            regions = load_litematic(path)

        self.assertEqual([region.name for region in regions], ["curve"])

    def test_a_schem_suffix_writes_a_schematic(self) -> None:
        with TemporaryDirectory() as directory:
            path = self.written(directory, "curve.schem")

            region, offset = load_schematic(path)

        self.assertTrue(region.has_content)
        # The top-level Offset retains the original world coordinates.
        self.assertEqual(offset, region.origin)

    def test_absolute_schematic_reconstructs_the_world_and_anchors_the_path(self) -> None:
        first = BlockPos(120, 75, -30)
        with TemporaryDirectory() as directory:
            path = self.written(directory, "absolute.schem", "absolute", first)
            region, corner = load_schematic(path)
            body = nbt.read(path).compound

        metadata = body["Metadata"]
        offset = BlockPos(*(metadata[f"WEOffset{axis}"].py_int for axis in "XYZ"))
        self.assertEqual(corner - offset, first)
        self.assertEqual(corner, region.origin)
        # WorldEdit's //paste -o keeps the saved world coordinates.
        self.assertTrue(any(position.y == first.y for position, _ in region.iter_occupied()))

    def test_relative_schematic_puts_first_path_point_at_player(self) -> None:
        first = BlockPos(120, 75, -30)
        with TemporaryDirectory() as directory:
            path = self.written(directory, "relative.schem", "relative", first)
            region, corner = load_schematic(path)
            body = nbt.read(path).compound
            absolute, _ = load_schematic(
                self.written(directory, "absolute.schem", "absolute", first)
            )

        metadata = body["Metadata"]
        offset = BlockPos(*(metadata[f"WEOffset{axis}"].py_int for axis in "XYZ"))
        self.assertEqual(corner - offset, BlockPos(0, 0, 0))
        self.assertEqual(
            {(position + first, state) for position, state in region.iter_occupied()},
            set(absolute.iter_occupied()),
        )

    def test_litematic_modes_use_the_first_path_point_as_reference(self) -> None:
        first = BlockPos(120, 75, -30)
        with TemporaryDirectory() as directory:
            relative = load_litematic(
                self.written(directory, "relative.litematic", "relative", first)
            )[0]
            absolute = load_litematic(
                self.written(directory, "absolute.litematic", "absolute", first)
            )[0]

        self.assertEqual(absolute.origin - relative.origin, first)

    def test_both_formats_hold_the_same_blocks(self) -> None:
        with TemporaryDirectory() as directory:
            litematic = load_litematic(self.written(directory, "curve.litematic"))[0]
            schematic, _ = load_schematic(self.written(directory, "curve.schem"))

        self.assertEqual(
            sorted(state_key(state) for _, state in litematic.iter_occupied()),
            sorted(state_key(state) for _, state in schematic.iter_occupied()),
        )


if __name__ == "__main__":
    unittest.main()
