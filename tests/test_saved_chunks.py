import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.Format import anvil, nbt
from src.Format.anvil import RegionFile, RegionPos, read_saved_chunk
from src.Terraform.region import RegionTerrain, WorldBlocks


def saved(directory: Path) -> Path:
    region = RegionFile(RegionPos(0, 0))
    region.chunks[(1, 2)] = nbt.CompoundTag({"Status": nbt.StringTag("minecraft:full")})
    return region.save(directory / "r.0.0.mca")


class SavedChunkTests(unittest.TestCase):
    def test_an_unfinished_chunk_is_not_available_for_survey_or_export(self) -> None:
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            region = RegionFile(RegionPos(0, 0))
            region.chunks[(1, 2)] = nbt.CompoundTag({
                "Status": nbt.StringTag("minecraft:structure_starts"),
            })
            region.save(directory / "r.0.0.mca")
            terrain, blocks = RegionTerrain(directory), WorldBlocks(directory)
            self.assertFalse(terrain.generated(1, 2))
            self.assertEqual(terrain.column(16, 32), ())
            self.assertFalse(blocks.generated(1, 2))
            self.assertIsNone(blocks.section(1, 4, 2))

    def test_a_saved_chunk_reads(self) -> None:
        with TemporaryDirectory() as folder:
            path = saved(Path(folder))
            self.assertEqual(read_saved_chunk(path, (1, 2))["Status"].py_str, "minecraft:full")
            self.assertIsNone(read_saved_chunk(path, (3, 3)))

    def test_a_chunk_the_game_is_still_writing_is_read_again_then_given_up(self) -> None:
        with TemporaryDirectory() as folder:
            path = saved(Path(folder))
            whole = path.read_bytes()
            # The chunk's sectors cut short, as mid-write.
            path.write_bytes(whole[: 2 * anvil.SECTOR + 12])
            with mock.patch.object(anvil, "SAVED_PAUSE", 0.0):
                with self.assertRaisesRegex(ValueError, "Close the world"):
                    read_saved_chunk(path, (1, 2))

    def test_a_chunk_finished_between_tries_is_read(self) -> None:
        with TemporaryDirectory() as folder:
            path = saved(Path(folder))
            whole = path.read_bytes()
            path.write_bytes(whole[: 2 * anvil.SECTOR + 12])

            def finish(_):
                path.write_bytes(whole)

            with mock.patch.object(anvil.time, "sleep", finish):
                self.assertEqual(read_saved_chunk(path, (1, 2))["Status"].py_str, "minecraft:full")


if __name__ == "__main__":
    unittest.main()
