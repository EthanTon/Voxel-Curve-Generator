from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from src.Auto.selector import (
    Choice, Facts, Rule, Selector, Settings, choose, crossing_lengths, fold, settle,
)
from src.Auto.composition import Composition, nearest_station
from src.Core.coordinates import BlockPos
from src.Assembly.structure import build_path, resolve_spans
from src.Parser import load


SAMPLE = Path(__file__).resolve().parents[1] / "sandbox" / "auto_assign"


def sample_settings(build_text=None, auto=None):
    data = tomllib.loads(build_text or (SAMPLE / "build.toml").read_text())
    if auto is not None:
        data["auto"] = auto
    return Settings.read(data, SAMPLE)


def test_section_rules_choose_from_terrain_facts():
    rules = sample_settings().rules

    def selected(y, ground, water=None, run=0):
        facts = Facts(y, ground, water, 1 if water is not None else 0, run, False)
        return choose(rules, facts, 0).section

    assert selected(70, 60) == "open_span"
    assert selected(70, 65) == "raised"
    assert selected(70, 70) == "plain"
    assert selected(70, 75) == "retained"
    assert selected(70, 84) == "enclosed"
    assert selected(70, 60, water=64, run=12) == "open_span"
    assert selected(65, 60, water=64, run=12) == "retained"
    assert selected(62, 60, water=64, run=12) == "retained"
    assert selected(55, 50, water=64, run=12) == "enclosed"
    assert selected(62, 60, water=64, run=100) == "enclosed"
    assert crossing_lengths([False, True, True, False, True]) == [0, 2, 2, 0, 1]


def test_equal_priority_between_sections_is_an_error():
    facts = Facts(65, 60, None, 0, 0, False)
    rules = (Rule("first", "first.rule", 10, 1, 0),
             Rule("second", "second.rule", 10, 1, 0))
    with pytest.raises(ValueError, match="ambiguous"):
        choose(rules, facts, 7)


def test_mrtsurface_water_cuts_do_not_select_or_fold_back_to_bridges():
    build = SAMPLE.parent / "mrtsurfacemap_bpath/build.toml"
    rules = Settings.read(tomllib.loads(build.read_text()), build.parent).rules
    # The reported river crossing is below water despite needing no fill.
    cut = choose(rules, Facts(59, 63, 62, 2, 30, False,
                             edge_ground=60, left_ground=63, right_ground=61), 1099)
    assert cut.section == "trench_water"
    bridge = choose(rules, Facts(70, 61, 62, 2, 30, False,
                                edge_ground=61, left_ground=61, right_ground=61), 0)
    assert bridge.section == "bridge_thin"
    # Water exactly at the formation also needs retaining walls.
    assert choose(rules, Facts(65, 58, 64, 2, 30, False,
                              edge_ground=58, left_ground=58, right_ground=58), 0).section == "trench_water"
    # Preserve deep covered tunnels and even one-station water cuts.
    assert choose(rules, Facts(43, 71, 70, 1, 4, False,
                              edge_ground=64, left_ground=73, right_ground=64), 0).section == "tunnel"
    route = [bridge] * 20 + [cut] + [bridge] * 20
    assert settle(route) == route


@pytest.mark.parametrize("water,ground,expected", [
    (63, 58, "embankment"),  # One block below the formation.
    (62, 58, "embankment"),
    (60, 58, "embankment"),
    (59, 56, "embankment"),  # Five blocks below; exactly eight blocks of fill.
    (58, 58, "bridge_thin"),  # Six blocks below.
    (62, 55, "bridge_thick"),  # Low clearance, but nine blocks of fill.
    (64, 58, "trench_water"),
    (65, 58, "trench_water"),
    (78, 58, "tunnel"),
    (None, 58, "embankment"),
])
def test_mrtsurface_low_clearance_water_uses_feasible_banks(water, ground, expected):
    build = SAMPLE.parent / "mrtsurfacemap_bpath/build.toml"
    settings = Settings.read(tomllib.loads(build.read_text()), build.parent)
    facts = Facts(65, ground, water, int(water is not None), 30, False,
                  edge_ground=ground, left_ground=ground, right_ground=ground)
    choice = choose(settings.rules, facts, 0)
    assert choice.section == expected
    if expected == "embankment" and water is not None:
        bridge = choose(settings.rules, Facts(65, 58, 58, 1, 30, False), 0)
        route = [bridge] * 20 + [choice] + [bridge] * 20
        assert settle(route) == route
        # A retaining half must preserve the bank's short-run protection too.
        mixed = settings.mixed["embankment"].apply(choice, Facts(
            65, ground, water, 1, 30, False, left_ground=72, right_ground=ground,
        ))
        assert mixed.section == "embankment_trench_left"
        assert fold([bridge] * 20 + [mixed] + [bridge] * 20)[20] == mixed


@pytest.mark.parametrize("pool_ground", [60, 75])
def test_lateral_water_survey_catches_higher_pool_outside_ground_samples(tmp_path, monkeypatch, pool_ground):
    from dataclasses import replace
    from src.Auto.selector import Column

    selector = make_selector(tmp_path)
    def column(x, z):
        water = 72 if z == 10 else 62 if z in (-3, 0, 3) else None
        return Column(x, z, pool_ground if z == 10 else 60, water, None, True)
    monkeypatch.setattr(selector, "column", column)
    original = selector.sample(3)
    selector.settings = replace(selector.settings, water_offset=12)
    extended = selector.sample(3)
    assert original[1] == 62
    assert extended[1] == 72
    assert extended[0] == original[0] == 60
    assert extended[3] == original[3] == False


def test_section_can_include_another_defined_section():
    build = (SAMPLE / "build.toml").read_text()
    build = build.replace("default = true\npriority = 0", "default = true\npriority = 0\ninclude = [\"raised\"]")
    settings = sample_settings(build, auto={"world": "/"})
    assert settings.includes["plain"] == ("raised",)


def test_dimension_follows_terraform_plan_unless_overridden(tmp_path):
    (tmp_path / "plan.toml").write_text(
        '[terraform]\nworld = "save"\ndimension = "minecraft:nova_proxima"\n'
    )
    plan = str(tmp_path / "plan.toml")
    assert sample_settings(auto={"world_from": plan}).dimension == "minecraft:nova_proxima"
    moon = {"world_from": plan, "dimension": "example:moon"}
    assert sample_settings(auto=moon).dimension == "example:moon"


def test_section_run_settings_preserve_short_water_crossings():
    plain = Choice("plain", "plain.default", 0, 20, 0)
    retained = Choice("retained", "retained.water", 100, 1, 0)
    route = [plain] * 25 + [retained] * 2 + [plain] * 25
    assert settle(route) == route
    assert settle([plain, plain, retained]) == [plain, plain, retained]


def test_a_one_station_side_variant_folds_into_its_neighbours():
    plain = Choice("plain", "plain.default", 0, 5, 0)
    left = Choice("plain_left", "plain.default:trench_left", 0, 5, 0, ("plain_left",))
    assert fold([plain] * 6 + [left] + [plain] * 6) == [plain] * 13
    assert fold([plain] * 6 + [left] * 5 + [plain] * 6) == [plain] * 6 + [left] * 5 + [plain] * 6


def test_sample_assignments_match_selected_stations_and_parse():
    selector = Selector.load(SAMPLE / "build.toml")
    if not selector.settings.world.exists():
        pytest.skip("sample Minecraft save is not installed")
    # A user's saved world changes after each paste, so the build selects afresh.
    spec = load(SAMPLE / "build.toml")
    assert len(spec.sections) == 5
    assert_resolution(selector, spec)


def test_selection_uses_the_same_terrain_baseline_as_terraform(tmp_path):
    (tmp_path / "plan.toml").write_text(
        '[terraform]\nworld = "target"\nsource_world = "original"\n'
    )
    settings = sample_settings(auto={"world_from": str(tmp_path / "plan.toml")})
    assert settings.world == tmp_path / "original"


def test_auto_and_authored_assignments_are_not_mixed(tmp_path):
    make_selector(tmp_path)
    build = tmp_path / "build.toml"
    build.write_text('path_assignments = "assignments.toml"\n' + build.read_text())
    with pytest.raises(ValueError, match="remove path_assignments"):
        load(build)


def assert_resolution(selector, spec=None):
    if spec is None:
        spec = load(selector.settings.base / "build.toml")
    resolved = resolve_spans(spec, build_path(spec))
    for index, station in enumerate(selector.stations):
        path_index = index + spec.path.lead
        actual = {name for name, spans in resolved.items()
                  if any(start <= path_index <= stop for start, stop in spans)}
        assert actual == set(station.selected_sections)


def make_selector(tmp_path, selection="", **options):
    write_build(tmp_path, selection, **options)
    return Selector.load(tmp_path / "build.toml")


def write_build(tmp_path, selection="", *, auto=True, world=False, points=None, lead=0,
                table="[auto]\noutput = \"assignments.toml\"\n"):
    points = points or [[i, 70, 0] for i in range(7)]
    sections = ""
    for name in ("plain", "raised", "retained", "enclosed"):
        sections += f'\n[[section]]\nname = "{name}"\ndomain = "bed"\nschematic = "{name}.schem"\n'
        if auto and name == "plain":
            sections += '[section.auto]\ndefault = true\nmin_run = 20\n'
        if auto and name == "raised":
            sections += ('[section.auto]\nexpand = 3\nmin_run = 20\n'
                         '[[section.auto.rule]]\npriority = 10\nrelief = { lt = -5 }\n')
    if world:
        table += 'world = "unused-world"\n'
    (tmp_path / "build.toml").write_text(
        table + '[output]\npath = "out.schem"\n'
        f'[path]\npoints = {points}\ngradients = {[[1, 0, 0]] * len(points)}\nlead = {lead}\n'
        '[marker]\nstart = [0, 0]\nend = [6, 0]\n[[domain]]\nname = "bed"\n'
        + sections + selection
    )


@pytest.mark.parametrize("modes", [
    {"auto"}, {"manual"}, {"control"}, {"auto", "manual"},
    {"auto", "control"}, {"manual", "control"}, {"auto", "manual", "control"},
])
def test_all_seven_mode_combinations(tmp_path, monkeypatch, modes):
    automatic = "auto" in modes
    selection = f'\n[selection]\nauto = {str(automatic).lower()}\n'
    expected = ["plain"] * 7
    if "manual" in modes:
        start = 2 if automatic else 0
        stop = 3 if automatic or "control" in modes else 6
        selection += f'[[selection.manual]]\nstations = [{start}, {stop}]\nsections = ["retained"]\n'
        expected[start:stop + 1] = ["retained"] * (stop - start + 1)
    if "control" in modes:
        start = 4 if automatic or "manual" in modes else 0
        # Deliberately author switches out of order and away from the route.
        selection += ('[[selection.control_point]]\npoint = [6, 80, 2]\nsections = ["enclosed"]\n'
                      f'[[selection.control_point]]\npoint = [{start}, 80, 2]\nsections = ["raised"]\n')
        expected[start:6] = ["raised"] * (6 - start)
        expected[6] = "enclosed"
    sampled = []
    monkeypatch.setattr(Selector, "sample", lambda self, i: sampled.append(i) or (70, None, 0, False))
    selector = make_selector(tmp_path, selection, auto=automatic, lead=2)
    assert [s.selected for s in selector.stations] == expected
    assert bool(sampled) == automatic
    assert_resolution(selector)


def test_manual_temporarily_overrides_control_and_switch_can_resume_auto(tmp_path, monkeypatch):
    selector = make_selector(tmp_path, '''
[selection]
[[selection.control_point]]
point = [1.1, 70, 3]
sections = ["raised"]
[[selection.control_point]]
point = [5, 70, 0]
auto = true
[[selection.manual]]
stations = [2, 2]
sections = ["enclosed"]
''')
    # Missing terrain under authored ranges must not block assignment.
    monkeypatch.setattr(Selector, "sample", lambda self, i: (70 if i in (0, 5, 6) else None, None, 0, False))
    assert [s.selected for s in selector.stations] == [
        "plain", "raised", "enclosed", "raised", "raised", "plain", "plain",
    ]
    assert [s.source for s in selector.stations] == [
        "auto", "control_point", "manual", "control_point", "control_point", "auto", "auto",
    ]
    assert selector.summary()["control_points"][0]["station"] == 1
    assert selector.point_detail([2.2, 70, 0])["source"] == "manual"
    assert_resolution(selector)


def test_manual_barrier_blocks_expansion_and_smoothing(tmp_path, monkeypatch):
    selector = make_selector(tmp_path, '''
[[selection.manual]]
stations = [2, 2]
sections = ["enclosed"]
''')
    monkeypatch.setattr(Selector, "sample", lambda self, i: (60 if i == 1 else 70, None, 0, False))
    assert [s.selected for s in selector.stations] == [
        "raised", "raised", "enclosed", "plain", "plain", "plain", "plain",
    ]
    assert_resolution(selector)


def test_switch_inside_manual_range_takes_effect_when_range_ends(tmp_path):
    selector = make_selector(tmp_path, '''
[selection]
auto = false
[[selection.control_point]]
point = [0, 0]
sections = ["plain"]
[[selection.control_point]]
point = [3, 0]
sections = ["raised"]
[[selection.manual]]
stations = [2, 4]
sections = ["enclosed"]
''', auto=False)
    assert [s.selected for s in selector.stations] == [
        "plain", "plain", "enclosed", "enclosed", "enclosed", "raised", "raised",
    ]
    assert_resolution(selector)


def test_full_authored_coverage_needs_no_rules_or_world_even_with_auto_enabled(tmp_path):
    selector = make_selector(tmp_path, '''
[[selection.manual]]
stations = [0, 6]
sections = ["plain"]
''', auto=False)
    assert selector.summary()["source_counts"] == {"manual": 7}
    assert selector.station_detail(0)["samples"] == []
    assert_resolution(selector)


def test_auto_includes_do_not_change_explicit_assignments(tmp_path, monkeypatch):
    selector = make_selector(tmp_path, '''
[[selection.manual]]
stations = [2, 2]
sections = ["plain"]
''')
    build = tmp_path / "build.toml"
    build.write_text(build.read_text().replace('default = true', 'default = true\ninclude = ["raised"]'))
    selector = Selector.load(build)
    monkeypatch.setattr(Selector, "sample", lambda self, i: (70, None, 0, False))
    assert selector.stations[0].selected_sections == ("plain", "raised")
    assert selector.stations[2].selected_sections == ("plain",)
    assert_resolution(selector)


def test_wet_crossing_length_continues_through_manual_assignment(tmp_path, monkeypatch):
    selector = make_selector(tmp_path, '''
[[selection.manual]]
stations = [2, 4]
sections = ["enclosed"]
''')
    monkeypatch.setattr(Selector, "sample", lambda self, i: (70, 71, 1, False))
    assert [s.water_run_points for s in selector.stations] == [7] * 7
    assert selector.stations[2].source == "manual"


def test_explicit_section_lists_are_kept_separate(tmp_path):
    selector = make_selector(tmp_path, '''
[selection]
auto = false
[[selection.manual]]
from = "start"
to = [2, 0]
sections = ["plain", "raised"]
[[selection.manual]]
stations = [3, 6]
sections = ["plain", "enclosed"]
''', auto=False)
    output = tomllib.loads(selector.assignments_toml())["assignment"]
    assert [a["sections"] for a in output] == [["plain", "raised"], ["plain", "enclosed"]]
    assert_resolution(selector)


def test_nearest_station_uses_height_when_provided_and_breaks_ties_by_index():
    points = [BlockPos(0, 0, 0), BlockPos(0, 100, 1)]
    assert nearest_station(points, [0, 100, 0])[0] == 1
    assert nearest_station(points, [0, 0])[0] == 0
    assert nearest_station(points, [0, 0.5])[0] == 0


def test_repeated_path_columns_and_single_station_runs_round_trip(tmp_path):
    selector = make_selector(tmp_path, '''
[selection]
auto = false
[[selection.control_point]]
point = [0, 70, 0]
sections = ["plain"]
[[selection.control_point]]
point = [0, 80, 0]
sections = ["enclosed"]
''', auto=False, points=[[0, 70, 0], [1, 75, 0], [0, 80, 0]])
    assert [s.selected for s in selector.stations] == ["plain", "plain", "enclosed"]
    assert_resolution(selector)


def test_terminal_switch_places_new_profile_at_final_world_point(tmp_path):
    from src.Assembly import build
    from src.Core.Field.block_field import BlockField
    from src.Core.block.palette import BlockPalette
    from src.Core.block.states import BlockState
    from src.Format.region import BlockRegion
    from src.Format.schematics import write_schematic

    selector = make_selector(tmp_path, '''
[selection]
auto = false
[[selection.control_point]]
station = 0
sections = ["plain"]
[[selection.control_point]]
station = -1
sections = ["enclosed"]
''', auto=False, lead=1)
    for name, block in (("plain", "minecraft:stone"), ("enclosed", "minecraft:gold_block")):
        field = BlockField(BlockPalette())
        field.set_block(BlockPos(0, 0, 0), BlockState.create(block))
        write_schematic(BlockRegion.from_field(field, name=name),
                        tmp_path / f"{name}.schem", data_version=3700)
    built = build(load(tmp_path / "build.toml"))
    assert built.world.get_block(BlockPos(5, 70, 0)).name == "minecraft:stone"
    assert built.world.get_block(BlockPos(6, 70, 0)).name == "minecraft:gold_block"


def test_index_switches_distinguish_repeated_columns_and_mix_with_points(tmp_path):
    selector = make_selector(tmp_path, '''
[selection]
auto = false
[[selection.control_point]]
station = -1
sections = ["enclosed"]
[[selection.control_point]]
point = [0, 70, 0]
sections = ["plain"]
[[selection.control_point]]
station = 1
sections = ["raised"]
''', auto=False, points=[[0, 70, 0], [1, 70, 0], [0, 70, 0]])
    assert [s.selected for s in selector.stations] == ["plain", "raised", "enclosed"]
    controls = selector.summary()["control_points"]
    assert [c["station"] for c in controls] == [0, 1, 2]
    assert controls[-1]["point"] == [0, 70, 0]
    assert controls[-1]["distance"] == 0
    assert_resolution(selector)


def test_index_switch_can_resume_auto(tmp_path, monkeypatch):
    selector = make_selector(tmp_path, '''
[[selection.control_point]]
station = 0
sections = ["raised"]
[[selection.control_point]]
station = -1
auto = true
''')
    monkeypatch.setattr(Selector, "sample", lambda self, i: (70, None, 0, False))
    assert [s.selected for s in selector.stations] == ["raised"] * 6 + ["plain"]
    assert_resolution(selector)


@pytest.mark.parametrize("stations,expected", [
    ([0, -1], (0, 6)), ([4, -1], (4, 6)), ([-1, -1], (6, 6)), ([-1, 6], (6, 6)),
])
def test_manual_last_station_ranges(tmp_path, stations, expected):
    selector = make_selector(tmp_path, f'''
[selection]
auto = false
[[selection.control_point]]
station = 0
sections = ["plain"]
[[selection.manual]]
stations = {stations}
sections = ["raised"]
''', auto=False, lead=2)
    start, stop = expected
    assert [s.selected for s in selector.stations] == [
        "raised" if start <= i <= stop else "plain" for i in range(7)
    ]
    assert_resolution(selector)


@pytest.mark.parametrize(("selection", "message"), [
    ({"auto": "false"}, "true or false"),
    ({"unknown": 1}, "unknown selection"),
    ({"manual": [{"stations": [0, 6], "sections": ["missing"]}]}, "defined section"),
    ({"manual": [{"stations": [2, 1], "sections": ["plain"]}]}, "inclusive indices"),
    ({"manual": [{"stations": [0, 7], "sections": ["plain"]}]}, "inclusive indices"),
    ({"manual": [{"stations": [False, 1], "sections": ["plain"]}]}, "inclusive indices"),
    ({"manual": [{"stations": [-2, -1], "sections": ["plain"]}]}, "inclusive indices"),
    ({"manual": [{"stations": [0, -2], "sections": ["plain"]}]}, "inclusive indices"),
    ({"manual": [{"stations": [-1, 2], "sections": ["plain"]}]}, "inclusive indices"),
    ({"manual": [{"stations": [0, 1], "from": [0, 0], "sections": ["plain"]}]}, "not both"),
    ({"control_point": [{"point": [1, float("nan"), 0], "sections": ["plain"]}]}, "finite"),
    ({"control_point": [{"station": -2, "sections": ["plain"]}]}, "nonnegative integer or -1"),
    ({"control_point": [{"station": 7, "sections": ["plain"]}]}, "from 0 to 6"),
    ({"control_point": [{"station": True, "sections": ["plain"]}]}, "nonnegative integer or -1"),
    ({"control_point": [{"station": 1.5, "sections": ["plain"]}]}, "nonnegative integer or -1"),
    ({"control_point": [{"station": "1", "sections": ["plain"]}]}, "nonnegative integer or -1"),
    ({"control_point": [{"station": 1, "point": [1, 0], "sections": ["plain"]}]}, "not both"),
    ({"control_point": [{"station": -1, "sections": ["plain"]},
                        {"point": [6, 0], "sections": ["raised"]}]}, "same station"),
    ({"control_point": [{"point": [1, 0], "sections": ["plain"], "auto": True}]}, "either"),
    ({"auto": False, "control_point": [{"point": [1, 0], "auto": True}]}, "cannot resume"),
    ({"control_point": [{"point": [1, 0], "sections": ["plain"]},
                        {"point": [1.1, 0], "sections": ["raised"]}]}, "same station"),
    ({"manual": [{"stations": [0, 2], "sections": ["plain"]},
                 {"stations": [2, 4], "sections": ["raised"]}]}, "conflicting manual"),
])
def test_bad_composition_is_rejected(selection, message):
    with pytest.raises(ValueError, match=message):
        Composition.read(selection, [BlockPos(i, 70, 0) for i in range(7)], {"plain", "raised"}, {})


def test_disabled_auto_reports_uncovered_station(tmp_path):
    selector = make_selector(tmp_path, '[selection]\nauto = false\n', auto=False)
    with pytest.raises(ValueError, match="covers route station 0"):
        selector.write()
    assert not selector.settings.output.exists()


def test_auto_reports_missing_rules_or_world(tmp_path):
    selector = make_selector(tmp_path, auto=False)
    with pytest.raises(ValueError, match="no section.auto rules"):
        selector.assignments_toml()
    selector = make_selector(tmp_path)
    with pytest.raises(ValueError, match="world or world_from"):
        selector.assignments_toml()


AUTHORED = 'path_assignments = "assignments.toml"\n'


@pytest.mark.parametrize("stations", [[-1, 2], [-2, 2], [0, -2], [2, 1], [False, 2], [1], [1, "2"]])
def test_generated_assignment_station_validation(tmp_path, stations):
    write_build(tmp_path, table=AUTHORED)
    (tmp_path / "assignments.toml").write_text(f'[[assignment]]\nstations = {stations}\nsections = ["plain"]\n'.replace("False", "false"))
    with pytest.raises((ValueError, tomllib.TOMLDecodeError)):
        spec = load(tmp_path / "build.toml")
        resolve_spans(spec, build_path(spec))


@pytest.mark.parametrize("lead", [0, 2])
@pytest.mark.parametrize("stations,expected", [
    ([0, -1], (0, 6)), ([4, -1], (4, 6)), ([-1, -1], (6, 6)), ([-1, 6], (6, 6)),
])
def test_authored_last_station_ranges_exclude_padding(tmp_path, lead, stations, expected):
    write_build(tmp_path, table=AUTHORED, lead=lead)
    (tmp_path / "assignments.toml").write_text(
        f'[[assignment]]\nstations = {stations}\nsections = ["plain"]\n'
    )
    spec = load(tmp_path / "build.toml")
    assert resolve_spans(spec, build_path(spec))["plain"] == (
        tuple(index + lead for index in expected),
    )


def test_generated_assignment_rejects_station_beyond_route(tmp_path):
    write_build(tmp_path, table=AUTHORED)
    (tmp_path / "assignments.toml").write_text('[[assignment]]\nstations = [0, 7]\nsections = ["plain"]\n')
    spec = load(tmp_path / "build.toml")
    with pytest.raises(ValueError, match="past the last authored"):
        resolve_spans(spec, build_path(spec))
