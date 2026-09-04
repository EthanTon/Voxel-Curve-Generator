from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Mapping

from .info import BlockInfo, NeighborRule, RotationRule, SolidFace
from .states import BlockState


VANILLA_VERSION = "26.2"


@dataclass(frozen=True, slots=True)
class VanillaBlockDefinition:
    name: str
    behavior: str
    properties: Mapping[str, tuple[str, ...]]
    default_properties: Mapping[str, str]
    tags: frozenset[str]

    def default_state(self) -> BlockState:
        return BlockState.create(self.name, dict(self.default_properties))


class VanillaCatalog:
    """Versioned block schemas and resolver metadata from Mojang's report."""

    def __init__(self, payload: Mapping[str, object]) -> None:
        self.version = str(payload["version"])
        raw_blocks = payload["blocks"]
        if not isinstance(raw_blocks, dict):
            raise ValueError("Vanilla catalog has no block map")
        self._blocks = raw_blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, name: str) -> bool:
        return name in self._blocks

    def names(self) -> Iterator[str]:
        return iter(self._blocks)

    def definition(self, name: str) -> VanillaBlockDefinition:
        try:
            raw = self._blocks[name]
        except KeyError:
            raise KeyError(
                f"Block {name!r} is not in the Minecraft {self.version} catalog"
            ) from None
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid catalog entry for {name!r}")
        return VanillaBlockDefinition(
            name=name,
            behavior=str(raw["type"]),
            properties={
                key: tuple(values)
                for key, values in raw.get("properties", {}).items()
            },
            default_properties=dict(raw.get("default", {})),
            tags=frozenset(raw.get("tags", ())),
        )

    def default_state(self, name: str) -> BlockState:
        return self.definition(name).default_state()

    def info(self, state: BlockState) -> BlockInfo:
        definition = self.definition(state.name)
        properties = definition.properties
        rotation = RotationRule.NONE
        if "axis" in properties:
            rotation |= RotationRule.AXIS
        if "facing" in properties:
            rotation |= RotationRule.FACING
        if "rotation" in properties:
            rotation |= RotationRule.ROTATION_16
        if all(direction in properties for direction in _CARDINALS):
            rotation |= RotationRule.CARDINAL
        if "orientation" in properties:
            rotation |= RotationRule.ORIENTATION
        if any(value.startswith("ascending_") for value in properties.get("shape", ())):
            rotation |= RotationRule.RAIL_SHAPE

        behavior = definition.behavior
        return BlockInfo(
            rotation=rotation,
            neighbor_rule=_NEIGHBOR_RULES.get(behavior, NeighborRule.NONE),
            solid_faces=self.solid_faces(state),
            center_faces=self._face_metadata(
                state, definition, "center_faces", "center_face_states"
            ),
            rigid_faces=self._face_metadata(
                state, definition, "rigid_faces", "rigid_face_states"
            ),
            replaceable=behavior in _REPLACEABLE_BEHAVIORS,
            behavior=behavior,
            connection_exception=bool(
                self._state_metadata_value(
                    state,
                    definition,
                    "connection_exception",
                    "connection_exception_states",
                    False,
                )
            ),
            flammable=bool(
                self._state_metadata_value(
                    state,
                    definition,
                    "flammable",
                    "flammable_states",
                    False,
                )
            ),
            signal_source=bool(
                self._state_metadata_value(
                    state,
                    definition,
                    "signal_source",
                    "signal_source_states",
                    False,
                )
            ),
            wall_cover=int(
                self._state_metadata_value(
                    state,
                    definition,
                    "wall_cover",
                    "wall_cover_states",
                    0,
                )
            ),
            tags=definition.tags,
        )

    def _face_metadata(
        self,
        state: BlockState,
        definition: VanillaBlockDefinition,
        default_key: str,
        states_key: str,
    ) -> SolidFace:
        fallback = int(self.solid_faces(state))
        return SolidFace(
            int(
                self._state_metadata_value(
                    state, definition, default_key, states_key, fallback
                )
            )
        )

    def solid_faces(self, state: BlockState) -> SolidFace:
        definition = self.definition(state.name)
        raw = self._blocks[state.name]
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid catalog entry for {state.name!r}")

        fallback = (
            int(SolidFace.ALL)
            if definition.behavior in _FULL_CUBE_BEHAVIORS
            else int(SolidFace.NONE)
        )
        mask = int(raw.get("faces", fallback))
        face_states = raw.get("face_states", {})
        if face_states:
            properties = dict(definition.default_properties)
            properties.update(dict(state.properties))
            key = ",".join(
                properties[name] for name in sorted(definition.properties)
            )
            mask = int(face_states.get(key, mask))
        return SolidFace(mask)

    def _state_metadata_value(
        self,
        state: BlockState,
        definition: VanillaBlockDefinition,
        default_key: str,
        states_key: str,
        fallback: object,
    ) -> object:
        raw = self._blocks[state.name]
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid catalog entry for {state.name!r}")
        value = raw.get(default_key, fallback)
        state_values = raw.get(states_key, {})
        if state_values:
            properties = dict(definition.default_properties)
            properties.update(dict(state.properties))
            key = ",".join(
                properties[name] for name in sorted(definition.properties)
            )
            value = state_values.get(key, value)
        return value

    def validate(self, state: BlockState) -> None:
        definition = self.definition(state.name)
        supplied = dict(state.properties)
        unknown = supplied.keys() - definition.properties.keys()
        if unknown:
            raise ValueError(
                f"Unknown properties for {state.name}: {sorted(unknown)}"
            )
        for key, value in supplied.items():
            allowed = definition.properties[key]
            if value not in allowed:
                raise ValueError(
                    f"Invalid {state.name}[{key}={value}]; expected one of {allowed}"
                )


@lru_cache(maxsize=1)
def vanilla_catalog() -> VanillaCatalog:
    path = Path(__file__).with_name("data") / "vanilla_26_2.json"
    with path.open(encoding="utf-8") as source:
        return VanillaCatalog(json.load(source))


_CARDINALS = ("north", "east", "south", "west")

_NEIGHBOR_RULES = {
    "fence": NeighborRule.FENCE,
    "iron_bars": NeighborRule.PANE,
    "stained_glass_pane": NeighborRule.PANE,
    "weathering_copper_bar": NeighborRule.PANE,
    "wall": NeighborRule.WALL,
    "fence_gate": NeighborRule.FENCE_GATE,
    "stair": NeighborRule.STAIR,
    "weathering_copper_stair": NeighborRule.STAIR,
    "rail": NeighborRule.RAIL,
    "powered_rail": NeighborRule.RAIL,
    "detector_rail": NeighborRule.RAIL,
    "redstone_wire": NeighborRule.REDSTONE,
    "tripwire": NeighborRule.TRIPWIRE,
    "trip_wire_hook": NeighborRule.TRIPWIRE_HOOK,
    "chorus_plant": NeighborRule.CHORUS,
    "glow_lichen": NeighborRule.MULTIFACE,
    "sculk_vein": NeighborRule.MULTIFACE,
    "multiface": NeighborRule.MULTIFACE,
    "vine": NeighborRule.VINE,
    "fire": NeighborRule.FIRE,
    "mossy_carpet": NeighborRule.MOSS_CARPET,
    "scaffolding": NeighborRule.SCAFFOLDING,
    "shelf": NeighborRule.SHELF,
    "pointed_dripstone": NeighborRule.POINTED_DRIPSTONE,
    "sulfur_spike": NeighborRule.POINTED_DRIPSTONE,
    "tinted_particle_leaves": NeighborRule.LEAVES,
    "untinted_particle_leaves": NeighborRule.LEAVES,
    "mangrove_leaves": NeighborRule.LEAVES,
    "chest": NeighborRule.CHEST,
    "trapped_chest": NeighborRule.CHEST,
    "copper_chest": NeighborRule.CHEST,
    "weathering_copper_chest": NeighborRule.CHEST,
    "attached_stem": NeighborRule.STEM,
    "stem": NeighborRule.STEM,
    "lantern": NeighborRule.HANGING,
    "weathering_lantern": NeighborRule.HANGING,
    "huge_mushroom": NeighborRule.HUGE_MUSHROOM,
    "grass": NeighborRule.SNOWY,
    "mycelium": NeighborRule.SNOWY,
    "snowy_dirt": NeighborRule.SNOWY,
    "kelp": NeighborRule.GROWING_PLANT,
    "kelp_plant": NeighborRule.GROWING_PLANT,
    "cave_vines": NeighborRule.GROWING_PLANT,
    "cave_vines_plant": NeighborRule.GROWING_PLANT,
    "twisting_vines": NeighborRule.GROWING_PLANT,
    "twisting_vines_plant": NeighborRule.GROWING_PLANT,
    "weeping_vines": NeighborRule.GROWING_PLANT,
    "weeping_vines_plant": NeighborRule.GROWING_PLANT,
    "hanging_moss": NeighborRule.HANGING_MOSS,
    "campfire": NeighborRule.CAMPFIRE,
    "bed": NeighborRule.MULTIBLOCK,
    "door": NeighborRule.MULTIBLOCK,
    "double_plant": NeighborRule.MULTIBLOCK,
    "small_dripleaf": NeighborRule.MULTIBLOCK,
    "tall_flower": NeighborRule.MULTIBLOCK,
    "tall_seagrass": NeighborRule.MULTIBLOCK,
}

_SUPPORT_BEHAVIORS = {
    "amethyst_cluster",
    "banner",
    "base_coral_fan",
    "bell",
    "button",
    "ceiling_hanging_sign",
    "coral_fan",
    "grindstone",
    "hanging_roots",
    "ladder",
    "lever",
    "pressure_plate",
    "weighted_pressure_plate",
    "redstone_torch",
    "redstone_wall_torch",
    "skull",
    "player_head",
    "wither_skull",
    "spore_blossom",
    "standing_banner",
    "standing_sign",
    "torch",
    "wall_banner",
    "wall_coral_fan",
    "wall_hanging_sign",
    "wall_sign",
    "wall_skull",
    "player_wall_head",
    "piglinwallskull",
    "wither_wall_skull",
    "wall_torch",
}

for _behavior in _SUPPORT_BEHAVIORS:
    _NEIGHBOR_RULES[_behavior] = NeighborRule.SUPPORT

# Families whose default collision/support shape is a complete cube. Partial
# blocks are deliberately conservative: callers can supply more precise faces
# for a particular state without accidentally creating false fence/pane arms.
_FULL_CUBE_BEHAVIORS = {
    "amethyst",
    "block",
    "bonemealable_feature_placer",
    "budding_amethyst",
    "colored_falling",
    "command",
    "concrete_powder",
    "coral",
    "crying_obsidian",
    "drop_experience",
    "grass",
    "half_transparent",
    "hay",
    "honey",
    "ice",
    "infested",
    "infested_rotated_pillar",
    "magma",
    "mud",
    "mycelium",
    "netherrack",
    "nylium",
    "powder_snow",
    "powered",
    "redstone_ore",
    "rooted_dirt",
    "rotated_pillar",
    "sand",
    "sculk",
    "slime",
    "snowy_dirt",
    "soul_sand",
    "sponge",
    "stained_glass",
    "tinted_glass",
    "transparent",
    "waterlogged_transparent",
    "weathering_copper_full",
    "wet_sponge",
}

_REPLACEABLE_BEHAVIORS = {
    "air",
    "dry_vegetation",
    "fire",
    "flower",
    "liquid",
    "nether_roots",
    "nether_sprouts",
    "short_dry_grass",
    "snow_layer",
    "tall_dry_grass",
    "tall_grass",
}
