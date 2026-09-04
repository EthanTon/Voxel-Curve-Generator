from .states import BlockState


BlockId = int


class BlockPalette:
    def __init__(self):
        self._states: list[BlockState] = []
        self._ids: dict[BlockState, BlockId] = {}
        self._frozen = False

    def add(self, state: BlockState) -> BlockId:
        if state in self._ids:
            return self._ids[state]

        if self._frozen:
            raise RuntimeError("Palette is frozen")

        block_id = len(self._states)

        self._states.append(state)
        self._ids[state] = block_id

        return block_id

    def get_id(self, state: BlockState) -> BlockId:
        if state not in self._ids:
            raise KeyError(
                f"BlockState {state} does not exist in palette"
            )

        return self._ids[state]

    def get_state(self, block_id: BlockId) -> BlockState:
        if block_id < 0 or block_id >= len(self._states):
            raise IndexError(
                f"BlockID {block_id} does not exist in palette"
            )

        return self._states[block_id]

    def freeze(self) -> None:
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __len__(self) -> int:
        return len(self._states)
