import unittest
from math import cos, radians, sin
from pathlib import Path
from tempfile import TemporaryDirectory

from src.Assembly import assemble
from src.Core.Feature.dependent_path import DependentFeaturePath, PathProjection, resolve_feature_paths
from src.Core.Feature.feature import DependentFeature, Feature
from src.Core.Feature.feature_path import FeaturePath, spacing_rules
from src.Core.Field.block_field import BlockField
from src.Core.Domain import Domain
from src.Core.coordinates import BlockPos, Vec2
from src.Core.modes import PlacementMode
from src.Parser import parse
from tests import test_assembly


def feature_path(positions, *, gradients=None, indices=None):
    path = FeaturePath(rules=())
    for number, position in enumerate(positions):
        path.add((position[0], 0, position[2]),
                 (1, 0, 0) if gradients is None else gradients[number],
                 180, number if indices is None else indices[number], position[1], layout=180)
    return path


def positions(path):
    return [(p.position.x, p.position.y + p.y_offset, p.position.z) for p in path]


class DependentPathTests(unittest.TestCase):
    def test_distance_or_turn_selects_source_points_and_preserves_metadata(self):
        source = feature_path([(n, 64 + n, 0) for n in range(9)], gradients=[
            (cos(radians(60 if n >= 2 else 0)), 0, sin(radians(60 if n >= 2 else 0)))
            for n in range(9)
        ])
        selected, = DependentFeaturePath('rail', spacing_rules(4, 45), include_end=True).derive({'rail': (source,)})
        self.assertEqual([p.path_index for p in selected], [0, 2, 6, 8])
        self.assertEqual(positions(selected), [(n, 64 + n, 0) for n in (0, 2, 6, 8)])
        self.assertEqual(selected[1].gradient, source[2].gradient)
        self.assertEqual(selected[1].layout, source[2].layout)
        self.assertEqual([p.point_index for p in selected], list(range(4)))

    def test_source_is_unchanged_and_end_is_optional(self):
        source = feature_path([(n, 64, 0) for n in range(6)])
        before = source.points
        selected, = DependentFeaturePath('rail', spacing_rules(4, 0)).derive({'rail': (source,)})
        self.assertEqual([p.path_index for p in selected], [0, 4])
        self.assertEqual(source.points, before)

    def test_endpoint_preserves_a_different_height_in_the_same_column(self):
        source = feature_path([(0, 64, 0), (0, 65, 0)], indices=[0, 0])
        selected, = DependentFeaturePath('rail', rules=(lambda kept, point: not len(kept),),
                                         include_end=True).derive({'rail': (source,)})
        self.assertEqual(positions(selected), [(0, 64, 0), (0, 65, 0)])

    def test_custom_rules_and_dependent_feature_protocol(self):
        source = feature_path([(n, 64, 0) for n in range(5)])
        dependency = DependentFeaturePath('rail', rules=(lambda path, point: point.path_index % 2 == 0,))
        feature = DependentFeature(blocks=BlockField(), placement_mode=PlacementMode.OVERWRITE,
                                   domain=Domain(0, 'test'), dependency=dependency)
        self.assertIsInstance(feature, Feature)
        selected, = feature.paths({'rail': (source,)})
        self.assertEqual([p.path_index for p in selected], [0, 2, 4])

    def test_spacing_restarts_at_support_gaps(self):
        first = feature_path([(n, 64, 0) for n in range(3)])
        second = feature_path([(n, 64, 0) for n in range(8, 12)], indices=range(8, 12))
        selected = DependentFeaturePath('rail', spacing_rules(3, 0), include_end=True).derive({'rail': (first, second)})
        self.assertEqual([[p.path_index for p in run] for run in selected], [[0, 2], [8, 11]])

    def test_minimum_distance_limits_turns_and_reserves_endpoint(self):
        source = feature_path([(n, 64, 0) for n in range(11)], gradients=[
            (cos(radians(n * 20)), 0, sin(radians(n * 20))) for n in range(11)
        ])
        selected, = DependentFeaturePath('rail', spacing_rules(10, 5), include_end=True,
                                         min_distance=3).derive({'rail': (source,)})
        self.assertEqual([p.path_index for p in selected], [0, 3, 6, 10])

    def test_short_run_keeps_both_ends_despite_minimum(self):
        source = feature_path([(n, 64, 0) for n in range(3)])
        selected, = DependentFeaturePath('rail', spacing_rules(10, 0), include_end=True,
                                         min_distance=3).derive({'rail': (source,)})
        self.assertEqual([p.path_index for p in selected], [0, 2])

    def test_chains_resolve_in_any_order(self):
        source = feature_path([(n, 64, 0) for n in range(8)])
        resolved = resolve_feature_paths({'rail': (source,)}, {
            'poles': DependentFeaturePath('anchors', rules=(lambda path, point: point.path_index >= 3,)),
            'anchors': DependentFeaturePath('rail', spacing_rules(3, 0)),
        })
        self.assertEqual([p.path_index for p in resolved['poles'][0]], [3, 6])

    def test_missing_sources_targets_and_cycles_are_errors(self):
        for graph, error in [
            ({'a': DependentFeaturePath('missing')}, 'unknown feature path'),
            ({'a': DependentFeaturePath('rail', projection=PathProjection('missing'))}, 'unknown feature path'),
            ({'a': DependentFeaturePath('b'), 'b': DependentFeaturePath('a')}, 'dependency cycle'),
            ({'a': DependentFeaturePath('rail', projection=PathProjection('a'))}, 'dependency cycle'),
        ]:
            with self.subTest(graph=graph), self.assertRaisesRegex(ValueError, error):
                resolve_feature_paths({'rail': ()}, graph)
        with self.assertRaisesRegex(ValueError, 'defined twice'):
            resolve_feature_paths({'rail': ()}, {'rail': DependentFeaturePath('rail')})


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.source = feature_path([(2, 64, 0)])[0]

    def test_left_ray_intersects_between_target_points_and_interpolates_height(self):
        target = feature_path([(0, 70, -4), (4, 74, -4)])
        hit = PathProjection('edge', max_distance=5).intersect(self.source, (target,))
        self.assertEqual(positions((hit,)), [(2, 72, -4)])
        self.assertEqual(hit.path_index, self.source.path_index)
        self.assertEqual(hit.gradient, self.source.gradient)

    def test_nearest_forward_hit_wins_regardless_of_target_order(self):
        targets = tuple(feature_path([(0, 64, z), (4, 64, z)]) for z in (-8, 3, -4))
        hit = PathProjection('edge', max_distance=10).intersect(self.source, targets)
        self.assertEqual(positions((hit,)), [(2, 64, -4)])

    def test_right_and_custom_direction(self):
        target = feature_path([(0, 64, 4), (4, 64, 4)])
        self.assertEqual(PathProjection('edge', angle=90).intersect(self.source, (target,)).position.z, 4)
        self.assertEqual(PathProjection('edge', direction=lambda point: Vec2(0, 1)).intersect(self.source, (target,)).position.z, 4)

    def test_intersection_can_be_a_condition_for_stamping_at_source(self):
        target = feature_path([(0, 70, -4), (4, 70, -4)])
        projection = PathProjection('edge', placement='source')
        self.assertEqual(projection.intersect(self.source, (target,)), self.source)
        self.assertIsNone(projection.intersect(self.source, ()))

    def test_segment_index_retains_sparse_crossings_and_stable_height_ties(self):
        source = feature_path([(55, 64, 0)])[0]
        first = feature_path([(n * 10, 70, -4) for n in range(20)])
        second = feature_path([(50, 60, -4), (60, 60, -4)])
        hit = PathProjection('edge').intersect(source, (first, second))
        self.assertEqual(positions((hit,)), [(55, 70, -4)])

    def test_custom_angle_uses_actual_gradient_including_reversed_travel(self):
        source = feature_path([(0, 64, 0)], gradients=[(-1, 0, 0)])[0]
        target = feature_path([(-5, 64, 5), (5, 64, 5)])
        hit = PathProjection('edge', angle=-45).intersect(source, (target,))
        self.assertEqual(positions((hit,)), [(-5, 64, 5)])

    def test_range_and_height_limits_can_reject_hits(self):
        target = feature_path([(0, 70, -4), (4, 70, -4)])
        for options in ({'max_distance': 3}, {'min_distance': 5}, {'height_tolerance': 2}):
            with self.subTest(options=options):
                self.assertIsNone(PathProjection('edge', **options).intersect(self.source, (target,)))

    def test_collinear_segment_singleton_and_endpoint_hits(self):
        projection = PathProjection('edge', min_distance=2)
        target = feature_path([(2, 64, -1), (2, 64, -5)])
        self.assertEqual(positions((projection.intersect(self.source, (target,)),)), [(2, 64, -2)])
        target = feature_path([(2, 64, -4)])
        self.assertEqual(projection.intersect(self.source, (target,)).position.z, -4)
        self.assertIsNone(projection.intersect(self.source, (feature_path([(3, 64, -4)]),)))

    def test_collinear_height_limit_selects_nearest_allowed_part(self):
        target = feature_path([(2, 70, -2), (2, 60, -12)])
        hit = PathProjection('edge', height_tolerance=1).intersect(self.source, (target,))
        self.assertEqual(positions((hit,)), [(2, 65, -7)])

    def test_vertical_target_can_be_met_at_source_height(self):
        target = feature_path([(2, 60, -4), (2, 70, -4)])
        hit = PathProjection('edge', height_tolerance=0).intersect(self.source, (target,))
        self.assertEqual(positions((hit,)), [(2, 64, -4)])

    def test_disjoint_target_runs_do_not_create_an_intersection_across_gap(self):
        targets = (feature_path([(0, 64, -4), (1, 64, -4)]),
                   feature_path([(3, 64, -4), (4, 64, -4)]))
        self.assertIsNone(PathProjection('edge').intersect(self.source, targets))

    def test_missed_projection_does_not_move_source_spacing(self):
        source = feature_path([(n, 64, 0) for n in range(7)])
        targets = tuple(feature_path([(n, 64, -4)]) for n in (1, 3, 6))
        selected, = DependentFeaturePath('rail', spacing_rules(3, 0), PathProjection('edge')).derive({'rail': (source,), 'edge': targets})
        self.assertEqual([p.path_index for p in selected], [3, 6])

    def test_invalid_projection_settings_and_vertical_heading_are_errors(self):
        for options in ({'max_distance': 0}, {'angle': float('nan')}, {'min_distance': -1},
                        {'height_tolerance': -1}, {'max_distance': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PathProjection('edge', **options)
        source = feature_path([(0, 64, 0)], gradients=[(0, 1, 0)])[0]
        with self.assertRaisesRegex(ValueError, 'horizontal direction'):
            PathProjection('edge').intersect(source, ())


class DependentAssemblyTests(unittest.TestCase):
    def configured(self, directory):
        data = test_assembly.StructureTests().structured(directory)
        data['domain'] += [{'name': 'poles', 'order': 2}, {'name': 'wire', 'order': 3}]
        data['lane'] = [{'name': 'track', 'domain': 'rail', 'sections': ['standard'],
                         'block': 'minecraft:rail', 'on': 1, 'off': 0, 'offset': [0, 1, 0]}]
        data['feature_path'] = [{'name': 'anchors', 'source': 'lane:track', 'spacing': 3, 'angle': 5}]
        data['structure'][0].update(domain='poles', feature_path='anchors', offset=[0, 2, 0])
        return data

    def blocks(self, world, name):
        return sorted(tuple(p) for p in world.iter_source_positions()
                      if world.get_block_id(p) is not None and world.registry.get_state(world.get_block_id(p)).name == name)

    def test_structure_follows_rail_path_height_with_shared_anchors(self):
        with TemporaryDirectory() as directory:
            world = assemble(parse(self.configured(directory), Path(directory)))
        self.assertEqual(self.blocks(world, 'minecraft:gold_block'), [(n, 67, 0) for n in (0, 3, 6, 8)])

    def test_overhanging_poles_stop_at_tunnel_for_dependent_and_ordinary_paths(self):
        for dependent in (True, False):
            with self.subTest(dependent=dependent), TemporaryDirectory() as directory:
                data = self.configured(directory)
                data['section'].append({**data['section'][0], 'name': 'tunnel'})
                data['assignment'] = [{'stations': [0, 3], 'sections': ['standard']},
                                      {'stations': [4, 8], 'sections': ['tunnel']}]
                test_assembly.SweepTests().profile(Path(directory) / 'schems', 'pillar',
                    blocks=[(0, 0, z) for z in range(-2, 3)], block='minecraft:gold_block')
                if not dependent:
                    data['feature_path'] = []
                    del data['structure'][0]['feature_path']
                    data['structure'][0].update(spacing=3)
                world = assemble(parse(data, Path(directory)))
            poles = self.blocks(world, 'minecraft:gold_block')
            self.assertTrue(poles)
            self.assertTrue(all(x <= 3 for x, y, z in poles), poles)
            self.assertTrue(any(x == 3 for x, y, z in poles), poles)

    def test_projected_structure_is_stamped_at_target_path(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['lane'].append({**data['lane'][0], 'name': 'shoulder', 'block': 'minecraft:stone', 'offset': [-1, 1, 0]})
            data['feature_path'][0]['project'] = {'target': 'lane:shoulder', 'direction': 'left', 'max_distance': 3, 'min_distance': 0.5}
            world = assemble(parse(data, Path(directory)))
        self.assertEqual(self.blocks(world, 'minecraft:gold_block'), [(n, 67, -1) for n in (0, 3, 6, 8)])

    def test_wires_and_wire_bound_poles_use_exact_same_anchors(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['wire'] = [{'name': 'contact', 'domain': 'wire', 'sections': ['standard'],
                             'geometry': 'line', 'schematic': 'schems/pillar.schem',
                             'feature_path': 'anchors', 'offset': [0, 3, 0]}]
            del data['structure'][0]['feature_path']
            data['structure'][0]['wire'] = 'contact'
            world = assemble(parse(data, Path(directory)))
        self.assertEqual(self.blocks(world, 'minecraft:gold_block'), sorted(
            [(n, 67, 0) for n in (0, 3, 6, 8)] + [(n, 68, 0) for n in range(9)]))

    def test_source_lane_absence_prevents_dependent_placement(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['section'].append({**data['section'][0], 'name': 'unused'})
            data['lane'][0]['sections'] = ['unused']
            world = assemble(parse(data, Path(directory)))
        self.assertEqual(self.blocks(world, 'minecraft:gold_block'), [])

    def test_section_handover_does_not_restart_shared_spacing(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['section'].append({**data['section'][0], 'name': 'other'})
            data['assignment'] = [{'stations': [0, 3], 'sections': ['standard']},
                                  {'stations': [4, 8], 'sections': ['other']}]
            data['lane'][0]['sections'] += ['other']
            data['structure'][0]['sections'] += ['other']
            world = assemble(parse(data, Path(directory)))
        self.assertEqual(self.blocks(world, 'minecraft:gold_block'), [(n, 67, 0) for n in (0, 3, 6, 8)])

    def test_three_pole_variants_consume_one_anchor_sequence_across_handovers(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            names = ['standard', 'thin_bridge', 'thick_bridge']
            data['path']['points'] = [[n, 64, 0] for n in range(21)]
            data['path']['gradients'] = [[1, 0, 0]] * 21
            data['marker']['end'] = [20, 0]
            data['section'] += [{**data['section'][0], 'name': name} for name in names[1:]]
            data['assignment'] = [
                {'stations': [0, 6], 'sections': ['standard']},
                {'stations': [7, 13], 'sections': ['thin_bridge']},
                {'stations': [14, 20], 'sections': ['thick_bridge']},
            ]
            data['lane'][0]['sections'] = names
            data['feature_path'][0]['spacing'] = 5
            template = data['structure'][0]
            data['structure'] = []
            for name, block in zip(names, ['gold_block', 'emerald_block', 'diamond_block']):
                test_assembly.profile_schematic(Path(directory) / 'schems', name,
                                                blocks=[(0, 0, 0)], block='minecraft:' + block)
                data['structure'].append({**template, 'name': name, 'sections': [name],
                                          'headings': {'0.0': 'schems/' + name + '.schem'}})
            world = assemble(parse(data, Path(directory)))
        self.assertEqual(self.blocks(world, 'minecraft:gold_block'), [(0, 67, 0), (5, 67, 0)])
        self.assertEqual(self.blocks(world, 'minecraft:emerald_block'), [(10, 67, 0)])
        self.assertEqual(self.blocks(world, 'minecraft:diamond_block'), [(15, 67, 0), (20, 67, 0)])

    def test_curved_rail_poles_and_wire_share_world_columns(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['path']['points'] = [[n, 64, 0] for n in range(5)] + [[4, 64, -n] for n in range(1, 5)]
            data['path']['gradients'] = [[1, 0, 0]] * 4 + [[1, 0, -1]] + [[0, 0, -1]] * 4
            data['marker']['end'] = [4, -4]
            data['wire'] = [{'name': 'contact', 'domain': 'wire', 'sections': ['standard'],
                             'geometry': 'line', 'schematic': 'schems/pillar.schem',
                             'feature_path': 'anchors', 'offset': [0, 3, 0]}]
            world = assemble(parse(data, Path(directory)))
        poles = [(x, y, z) for x, y, z in self.blocks(world, 'minecraft:gold_block') if y == 67]
        self.assertGreater(len(poles), 2)
        for x, y, z in poles:
            self.assertEqual(world.get_block((x, y - 2, z)).name, 'minecraft:rail')
            self.assertEqual(world.get_block((x, y + 1, z)).name, 'minecraft:gold_block')

    def test_droppers_connect_dependent_wire_heights(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['wire'] = [
                {'name': name, 'domain': 'wire', 'sections': ['standard'], 'geometry': 'line',
                 'schematic': 'schems/pillar.schem', 'feature_path': 'anchors', 'offset': [0, y, 0]}
                for name, y in (('contact', 3), ('messenger', 5))
            ]
            data['wire'].append({'name': 'droppers', 'domain': 'wire', 'between': ['contact', 'messenger'],
                                 'schematic': 'schems/pillar.schem', 'spacing': 3})
            world = assemble(parse(data, Path(directory)))
        for x in (0, 3, 6, 8):
            for y in range(68, 71):
                self.assertEqual(world.get_block((x, y, 0)).name, 'minecraft:gold_block')

    def test_named_path_chain_and_structure_source(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            # The second structure depends on already-offset pole origins.
            data['feature_path'].insert(0, {'name': 'tops', 'source': 'structure:pillar', 'include_end': False})
            data['structure'].append({**data['structure'][0], 'name': 'top', 'feature_path': 'tops', 'offset': [0, 1, 0]})
            world = assemble(parse(data, Path(directory)))
        self.assertEqual(self.blocks(world, 'minecraft:gold_block'), sorted(
            [(n, y, 0) for n in (0, 3, 6, 8) for y in (67, 68)]))

    def test_gaps_in_source_lane_restart_spacing_and_leave_gaps_unwired(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['section'].append({**data['section'][0], 'name': 'unwired'})
            data['assignment'] = [
                {'stations': [0, 2], 'sections': ['standard']},
                {'stations': [3, 5], 'sections': ['unwired']},
                {'stations': [6, 8], 'sections': ['standard']},
            ]
            data['wire'] = [{'name': 'contact', 'domain': 'wire', 'sections': ['standard'],
                             'geometry': 'line', 'schematic': 'schems/pillar.schem',
                             'feature_path': 'anchors', 'offset': [0, 3, 0]}]
            world = assemble(parse(data, Path(directory)))
        self.assertEqual(self.blocks(world, 'minecraft:gold_block'), sorted(
            [(n, 67, 0) for n in (0, 2, 6, 8)] + [(n, 68, 0) for n in (0, 1, 2, 6, 7, 8)]))

    def test_restricting_continuous_rail_source_reserves_exposed_endpoints(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['section'].append({**data['section'][0], 'name': 'tunnel'})
            data['assignment'] = [
                {'stations': [0, 2], 'sections': ['standard']},
                {'stations': [3, 5], 'sections': ['tunnel']},
                {'stations': [6, 8], 'sections': ['standard']},
            ]
            data['lane'][0]['sections'] = ['standard', 'tunnel']
            data['feature_path'][0]['sections'] = ['standard']
            world = assemble(parse(data, Path(directory)))
        self.assertEqual(self.blocks(world, 'minecraft:gold_block'), [(n, 67, 0) for n in (0, 2, 6, 8)])

    def test_followed_tunnel_wire_uses_every_dependent_rail_point(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['feature_path'][0].update(spacing=0, angle=0)
            data['wire'] = [{'name': 'contact', 'domain': 'wire', 'sections': ['standard'],
                             'geometry': 'follow', 'schematic': 'schems/pillar.schem',
                             'feature_path': 'anchors', 'offset': [0, 3, 0]}]
            data['structure'] = []
            world = assemble(parse(data, Path(directory)))
        self.assertEqual(self.blocks(world, 'minecraft:gold_block'), [(n, 68, 0) for n in range(9)])

    def test_parallel_dependent_catenary_does_not_cross_at_tunnel_handover(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['structure'] = []
            data['section'].append({**data['section'][0], 'name': 'tunnel'})
            data['assignment'] = [{'stations': [0, 3], 'sections': ['standard']},
                                  {'stations': [4, 8], 'sections': ['tunnel']}]
            data['lane'] = [{**data['lane'][0], 'name': f'rail_{side}', 'offset': [x, 1, 0],
                             'sections': ['standard', 'tunnel']} for side, x in (('left', -1), ('right', 1))]
            data['feature_path'] = [
                {'name': f'{section}_{side}', 'source': f'lane:rail_{side}', 'sections': [section],
                 'spacing': 3 if section == 'standard' else 0}
                for side in ('left', 'right') for section in ('standard', 'tunnel')
            ]
            data['wire'] = [
                {'name': f'{section}_{side}', 'domain': 'wire', 'sections': [section],
                 'geometry': 'line' if section == 'standard' else 'follow',
                 'schematic': 'schems/pillar.schem', 'feature_path': f'{section}_{side}', 'offset': [0, 3, 0]}
                for side in ('left', 'right') for section in ('standard', 'tunnel')
            ]
            world = assemble(parse(data, Path(directory)))
        wires = self.blocks(world, 'minecraft:gold_block')
        self.assertTrue(wires)
        self.assertTrue(all(z in (-1, 1) for x, y, z in wires))
        self.assertEqual({(x, z) for x, y, z in wires}, {(x, z) for x in range(9) for z in (-1, 1)})

    def test_diagonal_layout_join_does_not_restart_minimum_pole_spacing(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['section'][0]['width'] = 7
            test_assembly.SweepTests().profile(Path(directory) / 'schems',
                                               blocks=[(x, 0, 0) for x in range(-3, 4)])
            data['path']['points'] = [[n, 64, -n] for n in range(21)]
            data['path']['gradients'] = [[cos(radians(40 if n < 10 else 50)), 0,
                                         -sin(radians(40 if n < 10 else 50))] for n in range(21)]
            data['marker']['end'] = [20, -20]
            data['lane'][0]['offset'] = [-3, 1, 0]
            data['feature_path'][0].update(spacing=6, min_distance=4)
            world = assemble(parse(data, Path(directory)))
        poles = self.blocks(world, 'minecraft:gold_block')
        self.assertGreater(len(poles), 2)
        for (x, y, z), (a, b, c) in zip(poles, poles[1:]):
            self.assertGreaterEqual(((x - a) ** 2 + (z - c) ** 2) ** 0.5, 4)

    def test_center_triggers_place_both_poles_and_continue_on_rail_tunnel_paths(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['section'][0]['width'] = 7
            test_assembly.SweepTests().profile(Path(directory) / 'schems',
                                               blocks=[(x, 0, 0) for x in range(-3, 4)])
            data['section'].append({**data['section'][0], 'name': 'tunnel'})
            data['path'].update(lead=8, points=[[n, 64, 0] for n in range(101)], gradients=[
                [cos(radians(11 if 3 <= n < 8 else 0)), 0,
                 sin(radians(11 if 3 <= n < 8 else 0))] for n in range(101)
            ])
            data['marker']['end'] = [100, 0]
            data['assignment'] = [{'stations': [0, 50], 'sections': ['standard']},
                                  {'stations': [51, 100], 'sections': ['tunnel']}]
            data['lane'] = [{**data['lane'][0], 'name': f'rail_{side}', 'offset': [x, 1, 0],
                             'sections': ['standard', 'tunnel']} for side, x in (('left', -3), ('right', 3))]
            data['feature_path'] = [{'name': 'center', 'source': 'path', 'sections': ['standard'],
                                     'spacing': 31, 'angle': 10, 'min_distance': 5}]
            data['wire'] = []
            pole = data['structure'][0]
            data['structure'] = []
            for side in ('left', 'right'):
                data['feature_path'] += [
                    {'name': f'anchors_{side}', 'source': 'feature_path:center',
                     'project': {'target': f'lane:rail_{side}', 'direction': side}},
                    {'name': f'tunnel_{side}', 'source': f'lane:rail_{side}', 'sections': ['tunnel']},
                ]
                data['structure'].append({**pole, 'name': f'pole_{side}', 'feature_path': f'anchors_{side}'})
                data['wire'] += [
                    {'name': f'{section}_{side}', 'domain': 'wire', 'sections': [section],
                     'geometry': 'line' if section == 'standard' else 'follow',
                     'schematic': 'schems/pillar.schem', 'offset': [0, 3, 0],
                     'feature_path': f'anchors_{side}' if section == 'standard' else f'tunnel_{side}'}
                    for section in ('standard', 'tunnel')
                ]
            world = assemble(parse(data, Path(directory)))
        for side, z in (('left', -3), ('right', 3)):
            poles = [placement for placement in world.instances
                     if placement.transform.origin.y == 67 and placement.transform.origin.z == z]
            self.assertEqual([placement.sequence_index for placement in poles],
                             [8 + station for station in (0, 5, 10, 41, 50)], side)
            for placement in poles:
                origin = placement.transform.origin
                self.assertEqual(world.get_block(origin + (0, -2, 0)).name, 'minecraft:rail')
                self.assertEqual(world.get_block(origin + (0, 1, 0)).name, 'minecraft:gold_block')
        cells = {point for point in self.blocks(world, 'minecraft:gold_block') if point[1] == 68}
        self.assertEqual(test_assembly.pieces(cells, faces=True), 2)

    def test_diagonal_tunnel_handover_keeps_dependent_strand_face_connected(self):
        with TemporaryDirectory() as directory:
            data = self.configured(directory)
            data['structure'] = []
            data['section'][0].update(width=7, steps={'-3,0': 'manhattan'})
            test_assembly.SweepTests().profile(Path(directory) / 'schems',
                                               blocks=[(x, 0, 0) for x in range(-3, 4)])
            data['section'].append({**data['section'][0], 'name': 'tunnel'})
            data['path'].update(points=[[n, 64, -n] for n in range(21)], gradients=[[1, 0, -1]] * 21, lead=8)
            data['marker']['end'] = [20, -20]
            data['assignment'] = [{'stations': [0, 9], 'sections': ['standard']},
                                  {'stations': [10, 20], 'sections': ['tunnel']}]
            data['lane'][0].update(offset=[-3, 1, 0], sections=['standard', 'tunnel'])
            data['feature_path'] = [
                {'name': section, 'source': 'lane:track', 'sections': [section],
                 'spacing': 6 if section == 'standard' else 0}
                for section in ('standard', 'tunnel')
            ]
            data['wire'] = [
                {'name': section, 'domain': 'wire', 'sections': [section],
                 'geometry': 'line' if section == 'standard' else 'follow',
                 'schematic': 'schems/pillar.schem', 'feature_path': section, 'offset': [0, 3, 0]}
                for section in ('standard', 'tunnel')
            ]
            world = assemble(parse(data, Path(directory)))
        cells = set(self.blocks(world, 'minecraft:gold_block'))
        self.assertTrue(cells)
        self.assertEqual(test_assembly.pieces(cells, faces=True), 1)

    def test_parser_rejects_missing_paths_cycles_and_conflicting_rules(self):
        cases = [
            (lambda data: data['feature_path'][0].update(source='lane:missing'), 'unknown feature path'),
            (lambda data: data['feature_path'][0].update(source='structure:pillar'), 'dependency cycle'),
            (lambda data: data['feature_path'][0].update(project={'target': 'feature_path:anchors'}), 'dependency cycle'),
            (lambda data: data['feature_path'].append(data['feature_path'][0]), 'duplicate feature_path'),
            (lambda data: data['structure'][0].update(spacing=4), 'put spacing and angle'),
            (lambda data: data['feature_path'][0].update(spacing=float('inf')), 'spacing'),
            (lambda data: data['feature_path'][0].update(measure='index', spacing=1.5), 'whole number'),
            (lambda data: data['structure'][0].update(spacing=0), 'put spacing and angle'),
            (lambda data: data['feature_path'][0].update(project={'target': 'path', 'direction': 'up'}), 'direction'),
            (lambda data: data['feature_path'][0].update(project={'target': 'path', 'direction': 'left', 'angle': 45}), 'direction or angle'),
        ]
        with TemporaryDirectory() as directory:
            for change, error in cases:
                data = self.configured(directory)
                change(data)
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    parse(data, Path(directory))


if __name__ == '__main__':
    unittest.main()
