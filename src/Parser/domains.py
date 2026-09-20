from __future__ import annotations

from collections.abc import Mapping

from ..Core.Domain import ConnectivityRules, PlacementRules, TerrainRules
from ..Core.block.connectivity_options import ConnectionAction, ConnectionOverride
from ..Core.modes import PlacementMode, TerrainMaskMode
from .spec import DomainSpec, InteractionSpec
from .values import read_enum, read_properties, read_set, require


def read_domain(data: Mapping) -> DomainSpec:
    name = require(data, "name")
    return DomainSpec(
        name=name,
        parents=tuple(data.get("parents", ())),
        order=data.get("order", 0),
        properties=dict(data.get("properties", {})),
        connectivity=read_connectivity(data.get("connectivity", {}), name),
        placement=read_placement(data.get("placement", {})),
        terrain=read_terrain(data.get("terrain", {})),
    )


def read_connectivity(data: Mapping, name: str) -> ConnectivityRules:
    return ConnectivityRules(
        connects_to_self=data.get("connects_to_self"),
        connects_to_any=data.get("connects_to_any"),
        overrides=tuple(
            read_override(each, f"domain {name!r}")
            for each in data.get("override", ())
        ),
    )


def read_placement(data: Mapping) -> PlacementRules:
    return PlacementRules(
        mode=read_enum(PlacementMode, data.get("mode")),
        replace=data.get("replace"),
        protected=data.get("protected"),
    )


def read_terrain(data: Mapping) -> TerrainRules:
    return TerrainRules(
        is_terrain=data.get("is_terrain"),
        mask_mode=read_enum(TerrainMaskMode, data.get("mask_mode")),
    )


def read_interaction(data: Mapping) -> InteractionSpec:
    source, target = require(data, "source"), require(data, "target")
    where = f"interaction {source!r} to {target!r}"
    return InteractionSpec(
        source=source,
        target=target,
        connect=data.get("connect"),
        overwrite=data.get("overwrite"),
        overrides=tuple(
            read_override(each, where) for each in data.get("override", ())
        ),
        set_properties=data.get("set_properties"),
        properties=dict(data.get("properties", {})),
        grants=dict(data.get("grants", {})),
        symmetric=data.get("symmetric", True),
    )


def read_override(data: Mapping, where: str) -> ConnectionOverride:
    for scoped in ("source_domains", "neighbor_domains"):
        if scoped in data:
            raise ValueError(
                f"{where} sets {scoped} on an override; the domains it "
                "applies to are taken from where it is declared"
            )
    return ConnectionOverride(
        action=read_enum(ConnectionAction, data.get("action"), ConnectionAction.CONNECT),
        source_blocks=read_set(data.get("source_blocks")),
        neighbor_blocks=read_set(data.get("neighbor_blocks")),
        source_behaviors=read_set(data.get("source_behaviors")),
        neighbor_behaviors=read_set(data.get("neighbor_behaviors")),
        source_tags=read_set(data.get("source_tags")),
        neighbor_tags=read_set(data.get("neighbor_tags")),
        directions=read_set(data.get("directions")),
        source_properties=read_properties(data.get("source_properties")),
        neighbor_properties=read_properties(data.get("neighbor_properties")),
        value=data.get("value"),
        set_properties=data.get("set_properties"),
        symmetric=data.get("symmetric", False),
    )
