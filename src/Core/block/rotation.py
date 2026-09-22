from .states import BlockState

from ..constants import (
    CARDINAL_DIRECTIONS,
    MIRRORED_DIRECTIONS,
    MIRRORED_RAIL_SHAPES,
    RAIL_ROTATION,
)
from .info import BlockInfo, RotationRule

def rotate_block_state(
    state: BlockState,
    info: BlockInfo,
    quarter_turns: int,
) -> BlockState:

    quarter_turns %= 4

    if quarter_turns == 0:
        return state

    rules = info.rotation

    if rules == RotationRule.NONE:
        return state

    properties = dict(state.properties)

    if rules & RotationRule.AXIS:
        _rotate_axis(properties, quarter_turns)

    if rules & RotationRule.FACING:
        _rotate_facing(properties, quarter_turns)

    if rules & RotationRule.CARDINAL:
        _rotate_cardinals(properties, quarter_turns)

    if rules & RotationRule.ROTATION_16:
        _rotate_sign(properties, quarter_turns)

    if rules & RotationRule.RAIL_SHAPE:
        _rotate_rail_shape(properties, quarter_turns)

    if rules & RotationRule.ORIENTATION:
        _rotate_orientation(properties, quarter_turns)

    return BlockState.create(
        state.name,
        properties,
    )


def _rotate_direction(
    direction: str,
    quarter_turns: int,
) -> str:
    if direction not in CARDINAL_DIRECTIONS:
        return direction

    index = CARDINAL_DIRECTIONS.index(direction)
    return CARDINAL_DIRECTIONS[(index + quarter_turns) % 4]


def _rotate_axis(
    properties: dict[str, str],
    quarter_turns: int,
) -> None:
    if quarter_turns % 2 == 0:
        return

    axis = properties.get("axis")

    if axis == "x":
        properties["axis"] = "z"

    elif axis == "z":
        properties["axis"] = "x"


def _rotate_facing(
    properties: dict[str, str],
    quarter_turns: int,
) -> None:
    facing = properties.get("facing")

    if facing is None:
        return

    properties["facing"] = _rotate_direction(
        facing,
        quarter_turns,
    )


def _rotate_cardinals(
    properties: dict[str, str],
    quarter_turns: int,
) -> None:
    original = {
        direction: properties[direction]
        for direction in CARDINAL_DIRECTIONS
        if direction in properties
    }

    for direction in original:
        del properties[direction]

    for direction, value in original.items():
        new_direction = _rotate_direction(
            direction,
            quarter_turns,
        )

        properties[new_direction] = value


def _rotate_sign(
    properties: dict[str, str],
    quarter_turns: int,
) -> None:
    rotation = properties.get("rotation")

    if rotation is None:
        return

    properties["rotation"] = str(
        (int(rotation) + quarter_turns * 4) % 16
    )


def _rotate_rail_shape(
    properties: dict[str, str],
    quarter_turns: int,
) -> None:
    shape = properties.get("shape")

    if shape not in RAIL_ROTATION:
        return

    for _ in range(quarter_turns):
        shape = RAIL_ROTATION[shape]

    properties["shape"] = shape


def _rotate_orientation(
    properties: dict[str, str],
    quarter_turns: int,
) -> None:
    orientation = properties.get("orientation")

    if orientation is None:
        return

    first, second = orientation.split("_", 1)

    first = _rotate_direction(
        first,
        quarter_turns,
    )

    second = _rotate_direction(
        second,
        quarter_turns,
    )

    properties["orientation"] = (
        f"{first}_{second}"
    )


def mirror_block_state(
    state: BlockState,
    info: BlockInfo,
    axis: str,
) -> BlockState:
    # A block state reflected across the plane through the origin square to
    # `axis`, which is the world axis the mirror flips: "x" turns east into
    # west, "z" turns north into south.
    #
    # A reflection is not any number of turns, so a block that only knows how
    # to turn cannot be mirrored by turning it. Everything a turn moves --
    # which way a block faces, which sides it connects on, which way a rail
    # climbs -- a mirror moves too, and differently.
    if axis not in MIRRORED_DIRECTIONS:
        raise ValueError(f"A mirror flips x or z, not {axis!r}")

    rules = info.rotation
    if rules == RotationRule.NONE:
        return state

    swapped = MIRRORED_DIRECTIONS[axis]
    properties = dict(state.properties)

    # An axis is unmoved: a mirror across x leaves an x pillar an x pillar.
    if rules & RotationRule.FACING:
        facing = properties.get("facing")
        if facing is not None:
            properties["facing"] = swapped.get(facing, facing)

    if rules & RotationRule.CARDINAL:
        original = {
            direction: properties[direction]
            for direction in CARDINAL_DIRECTIONS
            if direction in properties
        }
        for direction in original:
            del properties[direction]
        for direction, value in original.items():
            properties[swapped.get(direction, direction)] = value

    if rules & RotationRule.ROTATION_16:
        _mirror_sign(properties, axis)

    if rules & RotationRule.RAIL_SHAPE:
        shape = properties.get("shape")
        if shape is not None:
            properties["shape"] = MIRRORED_RAIL_SHAPES[axis].get(shape, shape)

    if rules & RotationRule.ORIENTATION:
        orientation = properties.get("orientation")
        if orientation is not None:
            first, second = orientation.split("_", 1)
            properties["orientation"] = (
                f"{swapped.get(first, first)}_{swapped.get(second, second)}"
            )

    return BlockState.create(state.name, properties)


def _mirror_sign(properties: dict[str, str], axis: str) -> None:
    # Sixteen steps of yaw, 0 facing south and climbing clockwise. A mirror
    # across x reflects about the south-north line, and one across z about
    # the east-west line a quarter of the way round from it.
    rotation = properties.get("rotation")
    if rotation is None:
        return
    turned = int(rotation)
    properties["rotation"] = str((-turned if axis == "x" else 8 - turned) % 16)
