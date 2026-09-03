from states import BlockState

from Core.constants import RAIL_ROTATION, CARDINAL_DIRECTIONS, ROTATABLE_PROPERTIES

def rotate_block_state(
    block: BlockState,
    quarter_turns: int,
) -> BlockState:
    quarter_turns %= 4

    if quarter_turns == 0:
        return block

    properties = block.properties

    has_rotatable_property = (
        any(
            key in properties
            for key in ROTATABLE_PROPERTIES
        )
        or any(
            direction in properties
            for direction in CARDINAL_DIRECTIONS
        )
        or properties.get("shape") in RAIL_ROTATION
    )

    if not has_rotatable_property:
        return block

    properties = dict(properties)

    _rotate_axis(properties, quarter_turns)
    _rotate_facing(properties, quarter_turns)
    _rotate_cardinals(properties, quarter_turns)
    _rotate_sign(properties, quarter_turns)
    _rotate_rail_shape(properties, quarter_turns)
    _rotate_orientation(properties, quarter_turns)

    return BlockState.create(
        block.name,
        properties,
    )

def _rotate_direction(
    direction: str,
    quarter_turns: int,
) -> str:
    if direction not in CARDINAL_DIRECTIONS:
        return direction

    index = CARDINAL_DIRECTIONS.index(direction)

    return CARDINAL_DIRECTIONS[
        (index + quarter_turns) % 4
    ]


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