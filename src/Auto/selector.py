from __future__ import annotations

import json
import tomllib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from pathlib import Path
from statistics import median
from math import hypot
from typing import TypeVar

from ..Parser.paths import read_path
from ..Parser.sections import read_sections, with_definition_from_file
from ..Parser.values import listed_entries
from ..Terraform.classify import BlockClass, classify
from ..Terraform.terrain import load_terrain
from .composition import Composition, nearest_station, world_point

COMPARISONS = {"lt", "lte", "gt", "gte", "eq"}
MEASURES = {"relief", "edge_relief", "left_relief", "right_relief", "fill_required",
            "water_depth", "wet_samples", "water_run_points"}
T = TypeVar("T")


@dataclass(frozen=True)
class Facts:
    y: int
    ground: float | None
    water_surface: float | None
    wet_samples: int
    water_run_points: int
    structure: bool
    min_wet_samples: int = 1
    edge_ground: float | None = None
    left_ground: float | None = None
    right_ground: float | None = None
    formation_offset: int = -1

    @property
    def wet(self) -> bool:
        return self.wet_samples >= self.min_wet_samples

    def measure(self, name: str) -> float | int | None:
        return {
            "relief": None if self.ground is None else self.ground - self.y,
            "edge_relief": None if self.edge_ground is None else self.edge_ground - self.y,
            "left_relief": None if self.left_ground is None else self.left_ground - self.y,
            "right_relief": None if self.right_ground is None else self.right_ground - self.y,
            "fill_required": self.fill_required,
            "water_depth": None if not self.wet or self.water_surface is None
            else self.water_surface - self.y,
            "wet_samples": self.wet_samples,
            "water_run_points": self.water_run_points,
        }[name]

    @property
    def fill_required(self):
        ground = [v for v in (self.ground, self.edge_ground, self.left_ground, self.right_ground)
                  if v is not None]
        return max(0, self.y + self.formation_offset - min(ground)) if ground else None


@dataclass(frozen=True)
class MixedSides:
    left: str
    right: str
    both: str
    cut_threshold: float

    def apply(self, choice: Choice, facts: Facts) -> Choice:
        left = self.cut(facts.left_ground, facts.y)
        right = self.cut(facts.right_ground, facts.y)
        if left and right:
            side = "both"
        elif left or right:
            side = "left" if left else "right"
        else:
            return choice
        name = getattr(self, side)
        return replace(choice, section=name, sections=(name,), rule=f"{choice.rule}:trench_{side}")

    def cut(self, ground: float | None, y: int) -> bool:
        return ground is not None and ground - y > self.cut_threshold


def in_range(value: float | int | None, bounds: Mapping[str, float]) -> bool:
    if value is None:
        return False
    return all({
        "lt": lambda: value < limit,
        "lte": lambda: value <= limit,
        "gt": lambda: value > limit,
        "gte": lambda: value >= limit,
        "eq": lambda: value == limit,
    }[operator]() for operator, limit in bounds.items())


@dataclass(frozen=True)
class Rule:
    section: str
    label: str
    priority: int
    min_run: int
    expand: int
    water: str = "any"
    structure: bool | None = None
    measures: tuple[tuple[str, Mapping[str, float]], ...] = ()

    def matches(self, facts: Facts) -> bool:
        if self.water == "wet" and not facts.wet:
            return False
        if self.water == "dry" and facts.wet:
            return False
        if self.structure is not None and self.structure != facts.structure:
            return False
        return all(in_range(facts.measure(name), bounds) for name, bounds in self.measures)


@dataclass(frozen=True)
class Choice:
    section: str
    rule: str
    priority: int
    min_run: int
    expand: int
    sections: tuple[str, ...] = ()
    source: str = "auto"

    @property
    def section_names(self) -> tuple[str, ...]:
        return self.sections or (self.section,)


def choose(rules: tuple[Rule, ...], facts: Facts, index: int) -> Choice:
    matches = [rule for rule in rules if rule.matches(facts)]
    if not matches:
        raise ValueError(f"no section selection rule matches route station {index}")
    highest = max(rule.priority for rule in matches)
    winners = [rule for rule in matches if rule.priority == highest]
    names = {rule.section for rule in winners}
    if len(names) > 1:
        raise ValueError(
            f"section selection is ambiguous at station {index}: "
            f"priority {highest} matches {', '.join(sorted(names))}"
        )
    rule = winners[0]
    return Choice(rule.section, rule.label, rule.priority, rule.min_run, rule.expand)


def read_rule(section: str, raw: Mapping, defaults: Mapping, number: int) -> Rule:
    allowed = {"name", "priority", "min_run", "expand", "water", "structure", *MEASURES}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"section {section!r} has unknown auto rule fields: {sorted(extra)}")
    priority = raw.get("priority", defaults.get("priority", 0))
    min_run = raw.get("min_run", defaults.get("min_run", 20))
    expand = raw.get("expand", defaults.get("expand", 0))
    if not isinstance(priority, int) or isinstance(priority, bool):
        raise ValueError(f"section {section!r} rule priority must be an integer")
    if not isinstance(min_run, int) or isinstance(min_run, bool) or min_run < 1:
        raise ValueError(f"section {section!r} rule min_run must be a positive integer")
    if not isinstance(expand, int) or isinstance(expand, bool) or expand < 0:
        raise ValueError(f"section {section!r} rule expand must be a nonnegative integer")
    water = raw.get("water", "any")
    if water not in ("any", "wet", "dry"):
        raise ValueError(f"section {section!r} rule water must be any, wet, or dry")
    structure = raw.get("structure")
    if structure is not None and not isinstance(structure, bool):
        raise ValueError(f"section {section!r} rule structure must be true or false")
    measures = []
    for name in MEASURES:
        if name not in raw:
            continue
        bounds = raw[name]
        if not isinstance(bounds, Mapping) or not bounds or set(bounds) - COMPARISONS:
            raise ValueError(f"section {section!r} rule {name} needs lt/lte/gt/gte/eq bounds")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               for value in bounds.values()):
            raise ValueError(f"section {section!r} rule {name} bounds must be numbers")
        measures.append((name, bounds))
    label = raw.get("name", f"rule_{number}")
    if not isinstance(label, str) or not label:
        raise ValueError(f"section {section!r} rule name must be text")
    return Rule(section, f"{section}.{label}", priority, min_run, expand,
                water, structure, tuple(sorted(measures)))


@dataclass(frozen=True)
class Settings:
    base: Path
    world: Path | None
    output: Path | None
    rules: tuple[Rule, ...]
    includes: Mapping[str, tuple[str, ...]]
    sample_offset: int = 3
    min_wet_samples: int = 1
    dimension: str | None = None
    edge_offset: int | None = None
    mixed: Mapping[str, MixedSides] = field(default_factory=dict)
    formation_offset: int = -1
    # Optional lateral water survey, independent of the ground samples.
    water_offset: int | None = None

    @classmethod
    def read(cls, build_data: Mapping, base: Path) -> Settings:
        auto = build_data["auto"]
        if not isinstance(auto, Mapping):
            raise ValueError("auto must be a table")
        extra = set(auto) - {"world", "world_from", "dimension", "output", "survey"}
        if extra:
            raise ValueError(f"unknown auto options: {sorted(extra)}")
        world, dimension = read_world(auto, build_data, base)
        output = auto.get("output")
        if output is not None:
            output = (base / output).resolve()

        survey = auto.get("survey", {})
        extra_survey = set(survey) - {"sample_offset", "min_wet_samples", "edge_offset", "formation_offset", "water_offset"}
        if extra_survey:
            raise ValueError(f"unknown auto.survey options: {sorted(extra_survey)}")
        offset = survey.get("sample_offset", 3)
        wet = survey.get("min_wet_samples", 1)
        edge = survey.get("edge_offset")
        water_offset = survey.get("water_offset")
        formation = survey.get("formation_offset", -1)
        if not isinstance(formation, int) or isinstance(formation, bool):
            raise ValueError("auto.survey.formation_offset must be a whole local height")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
            raise ValueError("auto.survey.sample_offset must be a positive integer")
        if not isinstance(wet, int) or isinstance(wet, bool) or not 1 <= wet <= 9:
            raise ValueError("auto.survey.min_wet_samples must be 1..9")
        if edge is not None and (not isinstance(edge, int) or isinstance(edge, bool) or edge < 1):
            raise ValueError("auto.survey.edge_offset must be a positive integer")
        if water_offset is not None and (not isinstance(water_offset, int)
                                         or isinstance(water_offset, bool) or water_offset < 1):
            raise ValueError("auto.survey.water_offset must be a positive integer")

        defined = {section.name for section in read_sections(build_data, {}, base)}
        entries, section_base = listed_entries(build_data, "sections", "section", "sections", base)
        rules: list[Rule] = []
        mixed = {}
        includes: dict[str, tuple[str, ...]] = {name: () for name in defined}
        for entry in entries:
            section = with_definition_from_file(entry, section_base)
            name = section["name"]
            section_auto = section.get("auto")
            if section_auto is None:
                continue
            includes[name] = read_includes(name, section_auto, defined)
            if "mixed" in section_auto:
                mixed[name] = read_mixed(name, section_auto["mixed"], defined)
            rules.extend(read_section_rules(name, section_auto))
        if any(rule.section not in defined for rule in rules):
            raise ValueError("selection rules reference an undefined section")
        return cls(base, world, output, tuple(rules), includes,
                   offset, wet, dimension, edge, mixed, formation, water_offset)


def read_world(auto: Mapping, build_data: Mapping, base: Path) -> tuple[Path | None, str | None]:
    # A direct world wins; otherwise borrow the world of a terraform plan.
    dimension = auto.get("dimension")
    if "world" in auto:
        return (base / auto["world"]).resolve(), dimension
    plan_name = auto.get("world_from", build_data.get("terraform"))
    if plan_name is None:
        return None, dimension
    if not isinstance(plan_name, str):
        raise ValueError("auto.world_from must name a terraform file")
    plan_file = (base / plan_name).resolve()
    plan = tomllib.loads(plan_file.read_text()).get("terraform", {})
    named = plan.get("source_world") or plan.get("world")
    if not named:
        raise ValueError(f"no terraform.world in {plan_file}")
    return (plan_file.parent / named).resolve(), dimension or plan.get("dimension")


def read_includes(name: str, auto: Mapping, defined: set[str]) -> tuple[str, ...]:
    if not isinstance(auto, Mapping):
        raise ValueError(f"section {name!r} auto must be a table")
    extra = set(auto) - {"priority", "min_run", "expand", "default", "rule", "include", "mixed"}
    if extra:
        raise ValueError(f"section {name!r} has unknown auto fields: {sorted(extra)}")
    linked = auto.get("include", [])
    if (not isinstance(linked, list) or
            any(not isinstance(value, str) or value not in defined or value == name
                for value in linked) or len(linked) != len(set(linked))):
        raise ValueError(f"section {name!r} auto.include needs distinct, defined section names")
    return tuple(linked)


def read_mixed(name: str, raw: object, defined: set[str]) -> MixedSides:
    sides = ("left", "right", "both")
    if (not isinstance(raw, Mapping) or set(raw) != {*sides, "cut_threshold"}
            or any(raw[side] not in defined or raw[side] == name for side in sides)
            or not isinstance(raw["cut_threshold"], (int, float))
            or isinstance(raw["cut_threshold"], bool) or raw["cut_threshold"] < 0):
        raise ValueError(f"section {name!r} auto.mixed needs defined left/right/both sections and a nonnegative cut_threshold")
    return MixedSides(**raw)


def read_section_rules(name: str, auto: Mapping) -> list[Rule]:
    default = auto.get("default", False)
    if not isinstance(default, bool):
        raise ValueError(f"section {name!r} auto.default must be true or false")
    listed = auto.get("rule", [])
    if not isinstance(listed, list) or any(not isinstance(value, Mapping) for value in listed):
        raise ValueError(f"section {name!r} auto.rule must be a list of tables")
    if not default and not listed:
        raise ValueError(f"section {name!r} auto needs default or at least one rule")
    rules = [read_rule(name, {}, auto, 0)] if default else []
    rules.extend(read_rule(name, raw, auto, number) for number, raw in enumerate(listed, 1))
    return rules


@dataclass(frozen=True)
class Column:
    x: int
    z: int
    ground: int | None
    water: int | None
    structure: int | None
    generated: bool

    def as_dict(self) -> dict:
        return dict(x=self.x, z=self.z, ground=self.ground, water=self.water,
                    structure=self.structure, generated=self.generated)


@dataclass(frozen=True)
class Station:
    index: int
    x: int
    y: int
    z: int
    ground: float | None
    wet_samples: int
    water_surface: float | None
    water_run_points: int
    structure: bool
    raw: str
    raw_rule: str
    selected: str
    selected_sections: tuple[str, ...] = ()
    source: str = "auto"
    selected_rule: str = ""
    edge_ground: float | None = None
    left_ground: float | None = None
    right_ground: float | None = None
    fill_required: float | None = None

    def as_dict(self) -> dict:
        return dict(index=self.index, x=self.x, y=self.y, z=self.z,
                    ground=self.ground, relief=None if self.ground is None else self.ground - self.y,
                    edge_ground=self.edge_ground,
                    edge_relief=None if self.edge_ground is None else self.edge_ground - self.y,
                    left_ground=self.left_ground, right_ground=self.right_ground,
                    left_relief=None if self.left_ground is None else self.left_ground - self.y,
                    right_relief=None if self.right_ground is None else self.right_ground - self.y,
                    fill_required=self.fill_required,
                    wet_samples=self.wet_samples,
                    water_surface=self.water_surface,
                    water_depth=None if self.water_surface is None else self.water_surface - self.y,
                    water_run_points=self.water_run_points,
                    structure=self.structure, raw=self.raw, raw_rule=self.raw_rule,
                    selected=self.selected, sections=list(self.selected_sections),
                    source=self.source, selected_rule=self.selected_rule)


def runs(names: list[T]) -> list[tuple[T, int, int]]:
    result: list[tuple[T, int, int]] = []
    for index, name in enumerate(names):
        if result and result[-1][0] == name:
            result[-1] = name, result[-1][1], index + 1
        else:
            result.append((name, index, index + 1))
    return result


def crossing_lengths(wet: list[bool]) -> list[int]:
    lengths = [0] * len(wet)
    start = 0
    while start < len(wet):
        if not wet[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(wet) and wet[stop]:
            stop += 1
        lengths[start:stop] = [stop - start] * (stop - start)
        start = stop
    return lengths


def settle(raw: list[Choice]) -> list[Choice]:
    return fold(expand(raw))


# Strong selections claim up to `expand` lower or equal priority neighbours.
def expand(raw: list[Choice]) -> list[Choice]:
    chosen = list(raw)
    claims = [(choice.priority, 0, 0, -index) for index, choice in enumerate(raw)]
    for index, source in enumerate(raw):
        if source.expand == 0:
            continue
        for nearby in range(max(0, index - source.expand),
                            min(len(raw), index + source.expand + 1)):
            if source.priority < raw[nearby].priority:
                continue
            claim = (source.priority, 1, -abs(nearby - index), -index)
            if claim > claims[nearby]:
                claims[nearby] = claim
                chosen[nearby] = source
    return chosen


# Interior runs shorter than their min_run join their longer neighbour.
def fold(chosen: list[Choice]) -> list[Choice]:
    chosen = list(chosen)
    while True:
        grouped = runs([choice.section for choice in chosen])
        short = [
            (position, run) for position, run in enumerate(grouped)
            if len(grouped) > 1
            and run[2] - run[1] < min(choice.min_run for choice in chosen[run[1]:run[2]])
            and not (run[1] == 0 or run[2] == len(chosen))
        ]
        if not short:
            return chosen
        position, (_, start, stop) = min(short, key=lambda item: (item[1][2] - item[1][1], item[1][1]))
        neighbors = [grouped[i] for i in (position - 1, position + 1) if 0 <= i < len(grouped)]
        replacement = max(neighbors, key=lambda run: (run[2] - run[1], -run[1]))
        chosen[start:stop] = [chosen[replacement[1]]] * (stop - start)


class Selector:
    def __init__(self, settings: Settings, build_data: Mapping):
        self.settings = settings
        self.path = read_path(build_data["path"], settings.base)
        if len(self.path.points) < 2:
            raise ValueError("a route needs at least two path points")
        self.composition = Composition.read(
            build_data.get("selection", {}), self.path.points,
            set(settings.includes), build_data.get("marker", {}),
        )

    @classmethod
    def load(cls, build: Path | str) -> Selector:
        build = Path(build).resolve()
        data = tomllib.loads(build.read_text())
        if "auto" not in data:
            raise ValueError(f"{build} has no [auto] table")
        return cls(Settings.read(data, build.parent), data)

    @cached_property
    def terrain(self):
        if self.settings.world is None:
            raise ValueError("automatic selection and world inspection need world or world_from in [auto]")
        return load_terrain(self.settings.world, self.settings.dimension)

    @lru_cache(maxsize=32768)
    def column(self, x: int, z: int) -> Column:
        if not self.terrain.generated(x >> 4, z >> 4):
            return Column(x, z, None, None, None, False)
        ground = water = structure = None
        for y, state in self.terrain.column(x, z):
            kind = classify(state)
            if kind is BlockClass.GROUND:
                ground = y if ground is None else max(ground, y)
            elif kind is BlockClass.STRUCTURE:
                structure = y if structure is None else max(structure, y)
            if state.name in ("minecraft:water", "minecraft:bubble_column"):
                water = y if water is None else max(water, y)
        return Column(x, z, ground, water, structure, True)

    def sample(self, index: int) -> tuple[float | None, float | None, int, bool]:
        point = self.path.points[index]
        offset = self.settings.sample_offset
        samples = [self.column(point.x + dx, point.z + dz)
                   for dx in (-offset, 0, offset) for dz in (-offset, 0, offset)]
        levels = [item.ground for item in samples if item.ground is not None]
        wet_columns = {(item.x, item.z): item for item in samples}
        reach = self.settings.water_offset
        if reach is not None:
            # Retaining walls can expose a river just beyond the footprint.
            # Inspect every transverse column through the configured reach,
            # without changing the ground's median or the side-cut survey.
            gradient = self.path.gradients[index]
            dx, dz = gradient.x, gradient.z
            length = hypot(dx, dz)
            if length < 1e-9:
                before = self.path.points[max(0, index - 1)]
                after = self.path.points[min(len(self.path.points) - 1, index + 1)]
                dx, dz = after.x - before.x, after.z - before.z
                length = hypot(dx, dz)
            if length > 0:
                for distance in range(-reach, reach + 1):
                    x = round(point.x - distance * dz / length)
                    z = round(point.z + distance * dx / length)
                    wet_columns[x, z] = self.column(x, z)
        water = [item.water for item in wet_columns.values() if item.water is not None
                 and (item.ground is None or item.water >= item.ground
                      # A cut can also expose a pool under a bank's roof.
                      # Count buried water only when it can enter the bed.
                      or (reach is not None and item.water >= point.y + self.settings.formation_offset))]
        structure = any(item.structure is not None and item.structure >= item.ground
                        for item in samples if item.ground is not None)
        surface = (max(water) if reach is not None else median(water)) if water else None
        return median(levels) if levels else None, surface, len(water), structure

    def edge_ground(self, index: int) -> float | None:
        offset = self.settings.edge_offset
        if offset is None:
            return None
        point = self.path.points[index]
        levels = [column.ground for dx in (-offset, 0, offset)
                  for dz in (-offset, 0, offset)
                  if (column := self.column(point.x + dx, point.z + dz)).ground is not None]
        return min(levels) if levels else None

    def side_ground(self, index: int, side: int) -> float | None:
        point = self.path.points[index]
        gradient = self.path.gradients[index]
        length = hypot(gradient.x, gradient.z)
        if length < 1e-9:
            before = self.path.points[max(0, index - 1)]
            after = self.path.points[min(len(self.path.points) - 1, index + 1)]
            dx, dz = after.x - before.x, after.z - before.z
            length = hypot(dx, dz)
        else:
            dx, dz = gradient.x, gradient.z
        if length < 1e-9:
            return None
        levels = []
        for distance in range(max(1, self.settings.sample_offset - 1), self.settings.sample_offset + 2):
            x = round(point.x - side * distance * dz / length)
            z = round(point.z + side * distance * dx / length)
            column = self.column(x, z)
            if column.ground is not None:
                levels.append(column.ground)
        return median(levels) if levels else None

    @cached_property
    def stations(self) -> tuple[Station, ...]:
        overrides = self.composition.overrides
        automatic = [index for index, override in enumerate(overrides) if override is None]
        if automatic:
            if not self.composition.automatic:
                raise ValueError(f"no manual or control-point assignment covers route station {automatic[0]}")
            if not self.settings.rules:
                raise ValueError(f"station {automatic[0]} needs automatic selection, but the build defines no section.auto rules")
        # Survey the whole crossing when auto is used, even through authored spans.
        # Missing ground is only an error where a rule must select a section.
        measured = ([self.sample(index) for index in range(len(self.path.points))]
                    if automatic else [(None, None, 0, False)] * len(self.path.points))
        wet = [item[2] >= self.settings.min_wet_samples for item in measured]
        widths = crossing_lengths(wet)
        sides = bool(automatic) and (bool(self.settings.mixed) or any(
            name in {"left_relief", "right_relief", "fill_required"}
            for rule in self.settings.rules for name, _ in rule.measures
        ))
        facts = [Facts(point.y, ground, surface, count, widths[index], structure,
                       self.settings.min_wet_samples,
                       self.edge_ground(index) if automatic else None,
                       self.side_ground(index, -1) if sides else None,
                       self.side_ground(index, 1) if sides else None,
                       self.settings.formation_offset)
                 for index, (point, (ground, surface, count, structure))
                 in enumerate(zip(self.path.points, measured))]
        raw = []
        for index, (fact, override) in enumerate(zip(facts, overrides)):
            if override is not None:
                raw.append(Choice(override.sections[0], override.label, 0, 1, 0,
                                  override.sections, override.source))
                continue
            if fact.ground is None:
                point = self.path.points[index]
                raise ValueError(f"no ground near route station {index} at ({point.x}, {point.z})")
            choice = choose(self.settings.rules, fact, index)
            raw.append(Choice(choice.section, choice.rule, choice.priority, choice.min_run,
                              choice.expand, (choice.section, *self.settings.includes[choice.section])))
        selected = list(raw)
        # Authored boundaries are barriers to automatic expansion and smoothing.
        for source, start, stop in runs([choice.source for choice in raw]):
            if source == "auto":
                selected[start:stop] = settle(raw[start:stop])
        for index, choice in enumerate(selected):
            if choice.source == "auto" and choice.section in self.settings.mixed:
                selected[index] = self.settings.mixed[choice.section].apply(choice, facts[index])
        # A side variant chosen station by station must not flicker either.
        for source, start, stop in runs([choice.source for choice in selected]):
            if source == "auto":
                selected[start:stop] = fold(selected[start:stop])
        return tuple(Station(index, point.x, point.y, point.z, fact.ground,
                             fact.wet_samples, fact.water_surface, fact.water_run_points,
                             fact.structure, raw[index].section, raw[index].rule,
                             selected[index].section, selected[index].section_names,
                             selected[index].source, selected[index].rule, fact.edge_ground,
                             fact.left_ground, fact.right_ground, fact.fill_required)
                     for index, (point, fact) in enumerate(zip(self.path.points, facts)))

    def summary(self) -> dict:
        route = self.stations
        return {
            "world": str(self.settings.world) if self.settings.world is not None else None,
            "dimension": self.settings.dimension or "minecraft:overworld",
            "points": len(route), "raw_counts": dict(Counter(s.raw for s in route)),
            "selected_counts": dict(Counter(s.selected for s in route)),
            "source_counts": dict(Counter(s.source for s in route)),
            "control_points": [control.as_dict() for control in self.composition.controls],
            "runs": [dict(section=names[0], sections=list(names),
                          start=start, stop=stop - 1)
                     for names, start, stop in runs([s.selected_sections for s in route])],
        }

    def station_detail(self, index: int) -> dict:
        if not 0 <= index < len(self.path.points):
            raise ValueError(f"station must be from 0 to {len(self.path.points) - 1}")
        point = self.path.points[index]
        offset = self.settings.sample_offset
        samples = ([self.column(point.x + dx, point.z + dz).as_dict()
                    for dx in (-offset, 0, offset) for dz in (-offset, 0, offset)]
                   if self.settings.world is not None else [])
        edge = self.settings.edge_offset
        edge_samples = ([self.column(point.x + dx, point.z + dz).as_dict()
                         for dx in (-edge, 0, edge) for dz in (-edge, 0, edge)]
                        if edge is not None and self.settings.world is not None else [])
        return {**self.stations[index].as_dict(), "samples": samples,
                "edge_samples": edge_samples}

    def point_detail(self, point: list[float]) -> dict:
        point = world_point(point, "point")
        index, distance = nearest_station(self.path.points, point)
        return {"point": list(point), "distance": distance, **self.station_detail(index)}

    def column_detail(self, x: int, z: int) -> dict:
        summary = self.column(x, z).as_dict()
        stack = self.terrain.column(x, z)
        return {**summary, "blocks": [
            {"y": y, "name": state.name, "properties": dict(state.properties)}
            for y, state in stack[-64:]
        ], "truncated": len(stack) > 64}

    def assignments(self) -> list[dict]:
        return [{"stations": [start, stop - 1], "sections": list(names)}
                for names, start, stop in runs([s.selected_sections for s in self.stations])]

    def assignments_toml(self) -> str:
        lines = ["# Generated by src/Auto. Selection rules and sections are in the build.", ""]
        for assignment in self.assignments():
            lines.extend(["[[assignment]]", f"stations = {assignment['stations']}",
                          f"sections = {json.dumps(assignment['sections'], ensure_ascii=False)}", ""])
        return "\n".join(lines)

    def write(self) -> Path:
        target = self.settings.output
        if target is None:
            raise ValueError("set output in [auto] to write the assignments")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(self.assignments_toml())
        temporary.replace(target)
        return target
