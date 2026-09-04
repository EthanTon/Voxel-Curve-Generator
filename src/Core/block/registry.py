from __future__ import annotations

from typing import Callable

from .info import BlockInfo, DEFAULT_BLOCK_INFO
from .palette import BlockId, BlockPalette
from .states import BlockState


class BlockRegistry:
    def __init__(
        self,
        palette: BlockPalette | None = None,
        default_info: BlockInfo = DEFAULT_BLOCK_INFO,
        info_provider: Callable[[BlockState], BlockInfo] | None = None,
    ) -> None:
        self._palette = palette if palette is not None else BlockPalette()
        self._default_info = default_info
        self._info_provider = info_provider
        self._info: list[BlockInfo] = [default_info] * len(self._palette)

    @property
    def palette(self) -> BlockPalette:
        return self._palette

    @property
    def default_info(self) -> BlockInfo:
        return self._default_info

    @classmethod
    def for_vanilla_26_2(
        cls,
        palette: BlockPalette | None = None,
    ) -> "BlockRegistry":
        from .vanilla import vanilla_catalog

        return cls(palette=palette, info_provider=vanilla_catalog().info)

    def register_vanilla_defaults(self) -> dict[str, BlockId]:
        from .vanilla import vanilla_catalog

        catalog = vanilla_catalog()
        registered: dict[str, BlockId] = {}
        for name in catalog.names():
            state = catalog.default_state(name)
            registered[name] = self.register(state, catalog.info(state))
        return registered

    def register(
        self,
        state: BlockState,
        info: BlockInfo | None = None,
    ) -> BlockId:

        if info is None:
            info = (
                self._info_provider(state)
                if self._info_provider is not None
                else self._default_info
            )
        block_id = self._palette.add(state)
        self.set_info(block_id, info)
        return block_id

    def set_info(self, block_id: BlockId, info: BlockInfo) -> None:

        self._palette.get_state(block_id)
        self._sync()
        self._info[block_id] = info

    def get_info(self, block_id: BlockId) -> BlockInfo:

        self._palette.get_state(block_id)
        self._sync()
        return self._info[block_id]

    def get_info_for_state(self, state: BlockState) -> BlockInfo:
        return self.get_info(self._palette.get_id(state))

    def get_state(self, block_id: BlockId) -> BlockState:
        return self._palette.get_state(block_id)

    def get_id(self, state: BlockState) -> BlockId:
        return self._palette.get_id(state)

    def get_or_add_variant(
        self,
        source_id: BlockId,
        state: BlockState,
    ) -> BlockId:

        source_info = self.get_info(source_id)
        source_state = self.get_state(source_id)
        try:
            return self._palette.get_id(state)
        except KeyError:
            block_id = self._palette.add(state)
        if self._info_provider is not None and state.name != source_state.name:
            self.set_info(block_id, self._info_provider(state))
        else:
            self.set_info(block_id, source_info)
        return block_id

    def __len__(self) -> int:
        return len(self._palette)

    def _sync(self) -> None:
        missing = len(self._palette) - len(self._info)
        if missing > 0:
            self._info.extend([self._default_info] * missing)
