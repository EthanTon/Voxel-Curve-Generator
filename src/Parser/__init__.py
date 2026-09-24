from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path

from .assignments import read_assignments
from .domains import read_domain, read_interaction
from .fittings import read_lane, read_structure, read_wire
from .paths import read_markers, read_output, read_path
from .sections import read_sections
from .spec import (
    AssignmentSpec,
    BuildSpec,
    DomainSpec,
    InteractionSpec,
    LaneSpec,
    Measure,
    OutputSpec,
    SplitSpec,
    PathSpec,
    SectionElevation,
    SectionSpec,
    SpanSpec,
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
    "InteractionSpec",
    "LaneSpec",
    "Measure",
    "OutputSpec",
    "SplitSpec",
    "PathSpec",
    "SectionElevation",
    "SectionSpec",
    "SpanSpec",
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
            read_structure(each, base) for each in data.get("structure", ())
        ),
        wires=tuple(read_wire(each, base) for each in data.get("wire", ())),
        lanes=tuple(read_lane(each, base) for each in data.get("lane", ())),
        terraform=read_terraform(data, base),
        surface_world=read_surface_world(data.get("surface"), base),
    )
    check_references(spec)
    return spec


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
    if not isinstance(data, Mapping) or set(data) != {"world"}:
        raise ValueError("[surface] needs exactly one option: world")
    world = data["world"]
    if not isinstance(world, str) or not world:
        raise ValueError("[surface].world must name a world save or schematic")
    return base / world


def check_references(spec: BuildSpec) -> None:
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
