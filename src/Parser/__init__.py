from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path

from .assignments import read_assignments
from .domains import read_domain, read_interaction
from .fittings import read_lane, read_structure, read_wire
from .formations import read_formation
from .feature_paths import check_feature_paths, read_feature_path
from .paths import read_markers, read_output, read_path
from .sections import read_sections
from .spec import (
    AssignmentSpec,
    BuildSpec,
    DomainSpec,
    FeaturePathSpec,
    FormationSpec,
    InteractionSpec,
    LaneSpec,
    Measure,
    OutputSpec,
    SplitSpec,
    PathSpec,
    SectionElevation,
    SectionSpec,
    SpanSpec,
    StationSpanSpec,
    StructureSpec,
    WireGeometry,
    WireSpec,
)
from ..Terraform.plan import load as load_terraform
from .values import require

__all__ = [
    "AssignmentSpec",
    "BuildSpec",
    "DomainSpec",
    "FeaturePathSpec",
    "FormationSpec",
    "InteractionSpec",
    "LaneSpec",
    "Measure",
    "OutputSpec",
    "SplitSpec",
    "PathSpec",
    "SectionElevation",
    "SectionSpec",
    "SpanSpec",
    "StationSpanSpec",
    "StructureSpec",
    "WireGeometry",
    "WireSpec",
    "load",
    "parse",
]


def load(path: Path | str) -> BuildSpec:
    path = Path(path)
    return parse(tomllib.loads(path.read_text()), path.parent)


def parse(data: Mapping, base: Path = Path(".")) -> BuildSpec:
    markers = read_markers(data.get("marker", {}))
    feature_paths = tuple(read_feature_path(each) for each in data.get("feature_path", ()))
    named_paths = frozenset(each.name for each in feature_paths)
    spec = BuildSpec(
        output=read_output(require(data, "output"), base),
        path=read_path(require(data, "path"), base),
        markers=markers,
        sections=read_sections(data, markers, base),
        assignments=read_assignments(data, markers, base),
        domains=tuple(read_domain(each) for each in data.get("domain", ())),
        interactions=tuple(
            read_interaction(each) for each in data.get("interaction", ())
        ),
        structures=tuple(
            read_structure(each, base, named_paths) for each in data.get("structure", ())
        ),
        wires=tuple(read_wire(each, base) for each in data.get("wire", ())),
        lanes=tuple(read_lane(each, base) for each in data.get("lane", ())),
        feature_paths=feature_paths,
        terraform=read_terraform(data, base),
        surface_world=read_surface_world(data.get("surface"), base),
        surface_dimension=read_surface_dimension(data.get("surface")),
        composition=read_composition(data, base),
        formations=tuple(read_formation(each) for each in data.get("formation", ())),
    )
    check_references(spec)
    return spec


def read_composition(data: Mapping, base: Path) -> Path | None:
    named = data.get("composition")
    if named is None:
        return None
    if not isinstance(named, str) or not named:
        raise ValueError('composition must name a file: composition = "composition.toml"')
    return (base / named).resolve()


def read_terraform(data: Mapping, base: Path):
    named = data.get("terraform")
    if named is None:
        return None
    if isinstance(named, str):
        return load_terraform(base / named)
    raise ValueError(
        "terraforming is written in a file of its own: terraform = \"terraform.toml\""
    )


def read_surface_world(data: Mapping | None, base: Path) -> Path | None:
    if data is None:
        return None
    if not isinstance(data, Mapping) or not set(data) <= {"world", "dimension"}:
        raise ValueError("[surface] accepts world and dimension")
    world = data.get("world")
    if world is None:
        return None
    if not isinstance(world, str) or not world:
        raise ValueError("[surface].world must name a world save or schematic")
    return base / world


def read_surface_dimension(data: Mapping | None) -> str | None:
    if data is None:
        return None
    dimension = data.get("dimension")
    if dimension is not None and (not isinstance(dimension, str) or not dimension):
        raise ValueError("[surface].dimension must name a dimension")
    return dimension


def check_references(spec: BuildSpec) -> None:
    check_feature_paths(spec)
    if len({each.name for each in spec.formations}) != len(spec.formations):
        raise ValueError("formation names must be unique")
    for formation in spec.formations:
        for name in (formation.domain, *formation.rail_domains, *formation.replace_domains):
            spec.domain(name)
    for interaction in spec.interactions:
        spec.domain(interaction.source)
        spec.domain(interaction.target)
    for section in spec.sections:
        spec.domain(section.domain)
    for assignment in spec.assignments:
        for name in assignment.sections:
            spec.section(name)
    for domain in spec.domains:
        for parent in domain.parents:
            spec.domain(parent)
    for fitting in (*spec.structures, *spec.wires, *spec.lanes):
        spec.domain(fitting.domain)
        for name in fitting.sections:
            spec.section(name)
    if spec.terraform is not None:
        spec.domain(spec.terraform.domain)
        known = {each.name for each in (*spec.structures, *spec.lanes)}
        for follows in spec.terraform.follows:
            if follows.fitting not in known:
                listed = ", ".join(sorted(known)) or "none"
                raise ValueError(
                    f"terraforming follows {follows.fitting!r}, which is not a "
                    f"structure or lane. Known fittings: {listed}"
                )
    for wire in spec.wires:
        for name in wire.between:
            if spec.wire(name).between:
                raise ValueError(
                    f"wire {wire.name!r} stretches to {name!r}, which is "
                    "itself stretched between other wires"
                )
    for structure in spec.structures:
        if structure.wire is not None:
            wire = spec.wire(structure.wire)
            if wire.between or wire.geometry is not WireGeometry.LINE:
                raise ValueError(f"structure {structure.name!r} needs a line wire along sections")
            if not set(structure.sections) <= set(wire.sections):
                raise ValueError(f"structure {structure.name!r} must ride sections of wire {wire.name!r}")
            if (structure.offset.x, structure.offset.z) != (wire.offset.x, wire.offset.z):
                raise ValueError(f"structure {structure.name!r} must share its wire's across/along offset")
    shared: dict[str, StructureSpec] = {}
    for structure in spec.structures:
        if structure.feature_path is None or any(path.name == structure.feature_path for path in spec.feature_paths):
            continue
        first = shared.setdefault(structure.feature_path, structure)
        spacing = ("spacing", "angle", "measure", "min_distance")
        if any(getattr(structure, key) != getattr(first, key) for key in spacing):
            raise ValueError(
                f"structures {first.name!r} and {structure.name!r} share feature "
                f"path {structure.feature_path!r}, so need the same spacing, "
                "angle, measure and min_distance"
            )
