from collections.abc import Callable

from ..constants import (
    CARDINAL_DIRECTIONS,
    MIRRORED_DIRECTIONS,
    MIRRORED_RAIL_SHAPES,
    RAIL_ROTATION,
)
from .info import BlockInfo, RotationRule
from .states import BlockState

DirectionMap = Callable[[str], str]


def rotate_block_state(
    state: BlockState,
    info: BlockInfo,
    quarter_turns: int,
) -> BlockState:
    quarter_turns %= 4
    rules = info.rotation
    if quarter_turns == 0 or rules == RotationRule.NONE:
        return state

    def turn(direction: str) -> str:
        if direction not in CARDINAL_DIRECTIONS:
            return direction
        return CARDINAL_DIRECTIONS[(CARDINAL_DIRECTIONS.index(direction) + quarter_turns) % 4]

    properties = dict(state.properties)
    if rules & RotationRule.AXIS and quarter_turns % 2:
        axis = properties.get("axis")
        if axis in ("x", "z"):
            properties["axis"] = "z" if axis == "x" else "x"
    if rules & RotationRule.FACING:
        _map_facing(properties, turn)
    if rules & RotationRule.CARDINAL:
        _map_cardinals(properties, turn)
    if rules & RotationRule.ROTATION_16:
        rotation = properties.get("rotation")
        if rotation is not None:
            properties["rotation"] = str((int(rotation) + quarter_turns * 4) % 16)
    if rules & RotationRule.RAIL_SHAPE:
        shape = properties.get("shape")
        if shape in RAIL_ROTATION:
            for _ in range(quarter_turns):
                shape = RAIL_ROTATION[shape]
            properties["shape"] = shape
    if rules & RotationRule.ORIENTATION:
        _map_orientation(properties, turn)
    return BlockState.create(state.name, properties)


def mirror_block_state(
    state: BlockState,
    info: BlockInfo,
    axis: str,
) -> BlockState:
    # Reflects across the plane square to world `axis`: "x" swaps east and
    # west, "z" swaps north and south. An axis property is unchanged.
    if axis not in MIRRORED_DIRECTIONS:
        raise ValueError(f"A mirror flips x or z, not {axis!r}")
    rules = info.rotation
    if rules == RotationRule.NONE:
        return state

    swapped = MIRRORED_DIRECTIONS[axis]

    def swap(direction: str) -> str:
        return swapped.get(direction, direction)

    properties = dict(state.properties)
    if rules & RotationRule.FACING:
        _map_facing(properties, swap)
    if rules & RotationRule.CARDINAL:
        _map_cardinals(properties, swap)
    if rules & RotationRule.ROTATION_16:
        rotation = properties.get("rotation")
        if rotation is not None:
            # Sixteenths of a turn clockwise from south.
            turned = int(rotation)
            properties["rotation"] = str((-turned if axis == "x" else 8 - turned) % 16)
    if rules & RotationRule.RAIL_SHAPE:
        shape = properties.get("shape")
        if shape is not None:
            properties["shape"] = MIRRORED_RAIL_SHAPES[axis].get(shape, shape)
    if rules & RotationRule.ORIENTATION:
        _map_orientation(properties, swap)
    return BlockState.create(state.name, properties)


def _map_facing(properties: dict[str, str], move: DirectionMap) -> None:
    facing = properties.get("facing")
    if facing is not None:
        properties["facing"] = move(facing)


def _map_cardinals(properties: dict[str, str], move: DirectionMap) -> None:
    # Moves per-side properties (north=..., east=...) to their new sides.
    original = {
        direction: properties.pop(direction)
        for direction in CARDINAL_DIRECTIONS
        if direction in properties
    }
    for direction, value in original.items():
        properties[move(direction)] = value


def _map_orientation(properties: dict[str, str], move: DirectionMap) -> None:
    orientation = properties.get("orientation")
    if orientation is not None:
        first, second = orientation.split("_", 1)
        properties["orientation"] = f"{move(first)}_{move(second)}"
