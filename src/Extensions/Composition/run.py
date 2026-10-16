from __future__ import annotations

from dataclasses import replace
from functools import partial

from ...Assembly.assemble import Built, Laying, finalize_build, place_features, prepare_build
from ...Assembly.progress import Progress, report
from ...Core.block.registry import BlockRegistry
from .ownership import OwnedWorld
from .formations import lay_formations
from .parser import validate
from .paths import resolve_paths
from .regions import generate_region
from .spec import CompositionSpec
from .terrain import shape_terrain


def build(spec: CompositionSpec, *, progress: Progress | None = None) -> Built:
    validate(spec)
    world = OwnedWorld(BlockRegistry.for_vanilla_26_2(), spec.ownership)
    prepared = []
    deferred = bool(spec.feature_paths or spec.bindings)
    for component in spec.components:
        each = prepare_build(component.build, world=world if not prepared else None,
                             shared=prepared[0] if prepared else None, defer_features=deferred, progress=progress)
        prepared.append(each)
    if deferred:
        prepared = resolve_paths(spec, prepared)
    prepared = [replace(each, layings=tuple(laying for laying in each.layings if laying.feature not in component.suppressed))
                for component, each in zip(spec.components, prepared)]
    names = {id(each): component.name for component, each in zip(spec.components, prepared)}
    priorities = {id(each): (component.priority, component.name) for component, each in zip(spec.components, prepared)}
    world.domains = prepared[0].domains
    report(progress, "Preparing regions", 0, len(spec.regions))
    for index, region in enumerate(spec.regions, 1):
        field = generate_region(region, world.registry.palette)
        laying = Laying(region.domain, partial(world.domains.place, world, field, region.domain))
        each = replace(prepared[0], spec=replace(prepared[0].spec, terraform=None), layings=(laying,), formations={})
        prepared.append(each)
        names[id(each)] = region.name
        priorities[id(each)] = (0, region.name)
        report(progress, "Preparing regions", index, len(spec.regions))

    def select_component(component):
        world.component = names[id(component)]

    ordered = bool(spec.sources or any(each.priority for each in spec.components))
    try:
        placed = place_features(*prepared, before_place=select_component, defer_terrain=True,
                                priority=(lambda each: priorities[id(each)]) if ordered else None,
                                progress=progress)
    finally:
        world.component = None
    lay_formations(world, prepared[0].domains, spec.formations,
                   {names[id(each)]: each.formations for each in prepared})
    return finalize_build(placed, terrain_builder=partial(shape_terrain, spec=spec), progress=progress)
