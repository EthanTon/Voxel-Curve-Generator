import unittest
from dataclasses import replace
from math import cos, hypot, radians, sin
from pathlib import Path
from tempfile import TemporaryDirectory

from src.Assembly import assemble
from src.Assembly.structure import build_domains
from src.Assembly.anchors import anchor_path, structure_points, structure_runs, wire_routes, runs_under
from src.Assembly.assemble import sweep
from src.Assembly.structure import (
    build_path, resolve_spans, build_sections, build_schedules, build_path_frames,
)
from src.Generation.Geometry.frames import level_near
from src.Generation.Geometry.silhouette import PathSilhouette
from src.Stamping import laid_offset
from src.Core.Feature.feature_path import FeaturePath
from src.Core.Field.block_field import BlockField
from src.Core.Domain import ConnectivityRules, DomainRegistry
from src.Core.World.block_world import BlockWorld
from src.Core.block.registry import BlockRegistry
from src.Core.block.connectivity_options import ConnectionAction
from src.Core.block.states import BlockState
from src.Core.coordinates import BlockPos
from src.Format.region import BlockRegion
from src.Format.schematics import write_schematic
from src.Parser import load, parse
from tests import test_assembly


class WirePoleTests(unittest.TestCase):
    def configured(self, directory):
        data = test_assembly.WireTests().wired(directory, spacing=3, offset=[1, 3, 0])
        data['domain'].append({
            'name': 'pole', 'order': -1,
            'connectivity': {'connects_to_self': False, 'preserve_state': True},
        })
        data['structure'] = [{
            'name': 'pole', 'domain': 'pole', 'sections': ['standard'],
            'wire': 'catenary', 'offset': [1, 2, 0],
            'headings': {'0.0': 'schems/pole.schem'},
        }]
        data['wire'][0]['schematic'] = 'schems/bars.schem'
        data['interaction'] = [
            {'source': 'rail', 'target': 'pole', 'overwrite': False,
             'connect': False, 'symmetric': False, 'override': [{
                 'source_blocks': ['minecraft:iron_bars'],
                 'neighbor_blocks': ['minecraft:lever'], 'symmetric': False,
                 'max_distance': 1,
             }]},
            {'source': 'pole', 'target': 'rail', 'connect': False, 'symmetric': False},
        ]
        for name, position, state in [
            ('pole', (0, 1, 0), BlockState.create('minecraft:lever', {
                'face': 'floor', 'facing': 'north', 'powered': 'false',
            })),
            ('bars', (0, 0, 0), BlockState.create('minecraft:iron_bars', {
                'north': 'false', 'south': 'false', 'east': 'false',
                'west': 'false', 'waterlogged': 'false',
            })),
        ]:
            field = BlockField()
            field.set_block(position, state)
            region = BlockRegion.from_field(field, name=name)
            write_schematic(region, Path(directory) / f'schems/{name}.schem',
                            data_version=4671, offset=region.origin)
        return data

    def test_section_handovers_do_not_restart_pole_spacing(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['marker']['mid'] = [4, 0]
            data['section'].append({**data['section'][0], 'name': 'other'})
            data['assignment'] = [
                {'from': 'start', 'to': 'mid', 'sections': ['standard']},
                {'from': 'mid', 'to': 'end', 'sections': ['other']},
            ]
            data['wire'][0]['sections'] = ['standard', 'other']
            data['structure'][0]['sections'] = ['standard', 'other']
            world = assemble(parse(data, Path(directory)))
            levers = sorted(tuple(p) for p in world.iter_source_positions()
                            if world.get_block_id(p) is not None and
                            world.registry.get_state(world.get_block_id(p)).name == 'minecraft:lever')
            self.assertEqual(levers, [(x, 67, 1) for x in (0, 3, 6, 8)])
            for x in (1, 2, 4, 5, 7):
                bars = world.registry.get_state(world.get_block_id(BlockPos(x, 67, 1)))
                self.assertEqual(bars.get('east'), 'true')
                self.assertEqual(bars.get('west'), 'true')

    def test_connectivity_is_directional_and_wire_cannot_overwrite_pole(self):
        with TemporaryDirectory() as directory:
            spec = parse(self.configured(directory), Path(directory))
        policy = build_domains(spec).compile()
        bars = BlockState.create('minecraft:iron_bars')
        lever = BlockState.create('minecraft:lever')
        rail, pole = policy.id_of('rail'), policy.id_of('pole')
        self.assertFalse(policy.can_overwrite(rail, pole))
        self.assertEqual(policy.options.decide(bars, rail, lever, pole, 'east').action,
                         ConnectionAction.CONNECT)
        self.assertEqual(policy.options.decide(lever, pole, bars, rail, 'west').action,
                         ConnectionAction.DEFAULT)
        self.assertFalse(policy.can_connect(pole, rail))

    def test_independent_anchor_settings_are_refused(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['structure'][0]['spacing'] = 10
            with self.assertRaisesRegex(ValueError, "wire's anchors"):
                parse(data, Path(directory))

    def test_wire_does_not_connect_to_a_diagonal_lever(self):
        with TemporaryDirectory() as directory:
            domains = build_domains(parse(self.configured(directory), Path(directory)))
        registry = BlockRegistry.for_vanilla_26_2()
        world = BlockWorld(registry)
        domains.bind(world)
        lever = BlockState.create('minecraft:lever', {
            'face': 'floor', 'facing': 'north', 'powered': 'false',
        })
        pole = BlockField(registry.palette)
        pole.set_block((0, 3, 0), lever)
        domains.place(world, pole, 'pole')
        wire = BlockField(registry.palette)
        wire.set_block((-1, 3, -1), BlockState.create('minecraft:iron_bars'))
        domains.place(world, wire, 'rail')
        world.resolve_dirty(domains.resolver(registry))
        bars = registry.get_state(world.get_block_id((-1, 3, -1)))
        self.assertEqual(bars.get('east'), 'false')
        self.assertEqual(bars.get('south'), 'false')
        self.assertEqual(registry.get_state(world.get_block_id((0, 3, 0))), lever)

    def test_followed_wire_and_mismatched_offsets_are_refused(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['structure'][0]['offset'] = [0, 2, 0]
            with self.assertRaisesRegex(ValueError, 'across/along offset'):
                parse(data, Path(directory))
            data['structure'][0]['offset'] = [1, 2, 0]
            data['wire'][0]['geometry'] = 'follow'
            del data['wire'][0]['spacing']
            with self.assertRaisesRegex(ValueError, 'line wire'):
                parse(data, Path(directory))

    def test_preserving_authored_fittings_is_inherited_and_invalidates_cache(self):
        registry = BlockRegistry.for_vanilla_26_2()
        domains = DomainRegistry()
        parent = domains.create('authored', connectivity=ConnectivityRules(preserve_state=True))
        child = domains.create('pole', parents=[parent])
        world = BlockWorld(registry)
        domains.bind(world)
        field = BlockField(registry.palette)
        lever = BlockState.create('minecraft:lever', {
            'face': 'floor', 'facing': 'north', 'powered': 'false',
        })
        field.set_block((0, 1, 0), lever)
        domains.place(world, field, child)
        world.resolve_dirty(domains.resolver(registry))
        self.assertEqual(registry.get_state(world.get_block_id((0, 1, 0))), lever)
        epoch = world.policy_epoch
        domains.update(parent, connectivity=ConnectivityRules(preserve_state=False))
        world.resolve_dirty(domains.resolver(registry))
        self.assertIsNone(world.get_block_id((0, 1, 0)))
        self.assertGreater(world.policy_epoch, epoch)

    def test_terraform_route_has_two_continuous_wires_through_section_handovers(self):
        sample = Path(__file__).resolve().parents[1] / 'sandbox/terraform_test/build.toml'
        spec = replace(load(sample), terraform=None)
        world = assemble(spec)
        catenary = build_domains(spec).profile('catenary').domain_id
        cells, levers, rails = set(), set(), set()
        for position in world.iter_source_positions():
            block_id = world.get_block_id(position)
            if block_id is None:
                continue
            state = world.registry.get_state(block_id)
            if state.name == 'minecraft:iron_bars' and world.source_at(position).domain == catenary:
                cells.add(tuple(position))
            if state.name in ('minecraft:rail', 'minecraft:powered_rail'):
                rails.add((position.x, position.z))
            if state.name == 'minecraft:lever' and state.get('face') == 'floor':
                levers.add(tuple(position))
        self.assertTrue(levers)
        for x, y, z in cells:
            self.assertTrue(any((x + dx, z + dz) in rails
                                for dx in (-1, 0, 1) for dz in (-1, 0, 1)), (x, y, z))
        for x, y, z in levers:
            joined = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        near = (x + dx, y + dy, z + dz)
                        if dx * dx + dy * dy + dz * dz != 1 or dy != 0 or near not in cells:
                            continue
                        state = world.registry.get_state(world.get_block_id(near))
                        sides = ([] if dx == 0 else ['west' if dx > 0 else 'east'])
                        sides += ([] if dz == 0 else ['north' if dz > 0 else 'south'])
                        joined |= bool(sides) and all(state.get(side) == 'true' for side in sides)
            self.assertTrue(joined, (x, y, z))
        self.assertEqual(test_assembly.pieces(cells | levers, faces=True), 2)
        registry = BlockRegistry.for_vanilla_26_2()
        domains = build_domains(spec)
        path = build_path(spec)
        spans = resolve_spans(spec, path)
        sections = build_sections(spec, registry, domains, spans)
        schedules = build_schedules(sections, spans, len(path.path_points))
        frames = build_path_frames(spec, schedules, path)
        runs = sweep(spec, schedules, path, frames)
        level = level_near(frames)
        silhouette = PathSilhouette.for_sections(path, sections.values(), frames)
        for structure in spec.structures:
            if structure.wire is None:
                continue
            wire = spec.wire(structure.wire)
            for route in wire_routes(wire, runs_under(spec, wire, runs)):
                stamped = []
                for run, anchors in structure_runs(spec, structure, runs, (route,)):
                    for point in structure_points(structure, anchors, sections[run.section_name], level):
                        placed = point.position + laid_offset(structure.offset, point.laid_at, point.mirrored)
                        stamped.append(placed + (0, point.y_offset + 7, 0))
                self.assertEqual(len(stamped), sum(len(anchors) for _, anchors in route))
                self.assertEqual(len(set(stamped)), len(stamped))
                # Boundary anchors still drive the continuous wire, but a
                # pole's arm cannot occupy a section that has no poles.
                excluded = [silhouette.for_span(span)
                            for name, schedule in schedules.items() if name not in structure.sections
                            for span in schedule.spans]
                for point in stamped:
                    outside = not any(mask.contains(point.x, point.z) for mask in excluded)
                    self.assertEqual(tuple(point) in levers, outside, tuple(point))
                for before, after in zip(stamped, stamped[1:]):
                    self.assertGreaterEqual(hypot(after.x - before.x, after.z - before.z), wire.min_distance)
        for placement in world.instances:
            if placement.connectivity_domain != catenary:
                continue
            for point in world._footprints[placement.placement_id]:
                if world.source_at(point).domain != catenary:
                    state = world.registry.get_state(world.get_block_id(point))
                    self.assertEqual((state.name, state.get('face')), ('minecraft:lever', 'floor'))

    def test_minimum_distance_limits_turn_anchors_and_reserves_endpoint(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['wire'][0].update(spacing=10, min_distance=3, angle=5)
            wire = parse(data, Path(directory)).wires[0]
        path = FeaturePath(rules=())
        for index in range(11):
            angle = radians(index * 20)
            path.add((index, 0, 0), (cos(angle), 0, sin(angle)), 180, index, 0, layout=180)
        anchors = anchor_path(wire, path)
        self.assertEqual([point.path_index for point in anchors], [0, 3, 6, 10])

    def test_minimum_distance_removes_close_tail_pole_in_assembly(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['wire'][0]['min_distance'] = 3
            world = assemble(parse(data, Path(directory)))
            levers = sorted(tuple(point) for point in world.iter_source_positions()
                            if world.get_block_id(point) is not None and
                            world.registry.get_state(world.get_block_id(point)).name == 'minecraft:lever')
            self.assertEqual(levers, [(x, 67, 1) for x in (0, 3, 8)])

    def test_invalid_minimum_and_followed_wire_minimum_are_refused(self):
        with TemporaryDirectory() as directory:
            for minimum in (-1, '12', True, float('nan'), float('inf'), 4):
                data = self.configured(directory)
                data['wire'][0]['min_distance'] = minimum
                with self.assertRaisesRegex(ValueError, 'min_distance'):
                    parse(data, Path(directory))
            data = self.configured(directory)
            data['wire'][0].update(geometry='follow', min_distance=0)
            del data['wire'][0]['spacing']
            with self.assertRaisesRegex(ValueError, 'only place the anchors'):
                parse(data, Path(directory))

    def test_unwired_gap_starts_a_new_anchor_sequence(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['marker'].update(gap_start=[4, 0], gap_end=[6, 0])
            data['section'].append({**data['section'][0], 'name': 'unwired'})
            data['assignment'] = [
                {'from': 'start', 'to': 'gap_start', 'sections': ['standard']},
                {'from': 'gap_start', 'to': 'gap_end', 'sections': ['unwired']},
                {'from': 'gap_end', 'to': 'end', 'sections': ['standard']},
            ]
            world = assemble(parse(data, Path(directory)))
            levers = sorted(tuple(point) for point in world.iter_source_positions()
                            if world.get_block_id(point) is not None and
                            world.registry.get_state(world.get_block_id(point)).name == 'minecraft:lever')
            self.assertEqual(levers, [(x, 67, 1) for x in (0, 3, 6, 8)])
            for x in (4, 5):
                self.assertIsNone(world.get_block_id((x, 67, 1)))

    def test_independent_structure_can_set_a_minimum_distance(self):
        with TemporaryDirectory() as directory:
            data = test_assembly.StructureTests().structured(directory, spacing=3, min_distance=3)
            world = assemble(parse(data, Path(directory)))
            poles = sorted(tuple(point) for point in world.iter_source_positions()
                           if world.get_block_id(point) is not None and
                           world.registry.get_state(world.get_block_id(point)).name == 'minecraft:gold_block')
        self.assertEqual(poles, [(x, 64, 0) for x in (0, 3, 8)])

    def test_wire_detours_around_a_stamped_pole_arm_and_preserves_its_lever(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            field = BlockField()
            field.set_block((0, 1, 0), BlockState.create('minecraft:lever', {
                'face': 'floor', 'facing': 'north', 'powered': 'false',
            }))
            field.set_block((0, 1, 1), BlockState.create('minecraft:pale_oak_fence_gate', {
                'facing': 'north', 'open': 'true', 'in_wall': 'false', 'powered': 'false',
            }))
            region = BlockRegion.from_field(field, name='pole')
            write_schematic(region, Path(directory) / 'schems/pole.schem',
                            data_version=4671, offset=region.origin)
            world = assemble(parse(data, Path(directory)))
        state = world.registry.get_state(world.get_block_id((2, 67, 1)))
        self.assertEqual(state.name, 'minecraft:pale_oak_fence_gate')
        cells = set()
        for point in world.iter_source_positions():
            block_id = world.get_block_id(point)
            if block_id is not None and world.registry.get_state(block_id).name in ('minecraft:iron_bars', 'minecraft:lever'):
                cells.add(tuple(point))
        self.assertEqual(test_assembly.pieces(cells, faces=True), 1)
        self.assertTrue(any(z != 1 for _, _, z in cells))


if __name__ == '__main__':
    unittest.main()
