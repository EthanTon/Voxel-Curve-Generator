import pytest

from src.Core.block.states import BlockState
from src.Format import nbt


@pytest.mark.parametrize("tag, expected", [
    (nbt.StringTag("minecraft:stone"), BlockState.create("minecraft:stone")),
    (nbt.StringTag("minecraft:oak_log"), BlockState.create("minecraft:oak_log", {"axis": "y"})),
    (nbt.StringTag("oak_log"), BlockState.create("minecraft:oak_log", {"axis": "y"})),
    (nbt.StringTag("example:custom_block"), BlockState.create("example:custom_block")),
    (nbt.CompoundTag({"": nbt.StringTag("minecraft:oak_log")}),
     BlockState.create("minecraft:oak_log", {"axis": "y"})),
    (nbt.CompoundTag({"id": nbt.StringTag("minecraft:oak_log"),
                      "properties": nbt.CompoundTag({"axis": nbt.StringTag("x")})}),
     BlockState.create("minecraft:oak_log", {"axis": "x"})),
    (nbt.CompoundTag({"id": nbt.StringTag("minecraft:rail"),
                      "properties": nbt.CompoundTag({"shape": nbt.StringTag("east_west")})}),
     BlockState.create("minecraft:rail", {"shape": "east_west", "waterlogged": "false"})),
    (nbt.CompoundTag({"id": nbt.StringTag("minecraft:oak_log")}),
     BlockState.create("minecraft:oak_log", {"axis": "y"})),
    (nbt.CompoundTag({"id": nbt.StringTag("example:custom_block"),
                      "properties": nbt.CompoundTag({"active": nbt.StringTag("true")})}),
     BlockState.create("example:custom_block", {"active": "true"})),
    (nbt.CompoundTag({"Name": nbt.StringTag("minecraft:oak_log")}),
     BlockState.create("minecraft:oak_log")),
    (nbt.CompoundTag({"Name": nbt.StringTag("minecraft:oak_log"),
                      "Properties": nbt.CompoundTag({"axis": nbt.StringTag("z")})}),
     BlockState.create("minecraft:oak_log", {"axis": "z"})),
])
def test_block_state_formats_read_and_export_without_losing_properties(tag, expected):
    # Exercise the actual binary NBT reader, then the legacy schematic writer.
    root = nbt.loads(nbt.dumps(nbt.CompoundTag({"state": tag}))).compound
    state = nbt.block_state_from_tag(root["state"])
    assert state == expected
    assert nbt.block_state_from_tag(nbt.block_state_tag(state)) == expected
