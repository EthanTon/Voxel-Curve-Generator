from dataclasses import dataclass
from states import BlockState

BlockId = int

@dataclass
class BlockPalette:
    def __init__(self):
        self._states: list[BlockState] = []
        self._ids = dict[BlockState, int] = {}
        self._frozen = False
        
    def add(self,state: BlockState) -> BlockId:
        if self._frozen:
            raise RuntimeError(f"Palette is frozen")
        
        if state not in self._ids:
            block_id = len(self._states)
            self._states.append(state)
            self._ids[state] = block_id
        
        else: block_id = self.get_id(BlockState)
        
        return block_id
        
    def get_id(self, state: BlockState) -> BlockId:
        return self._ids.get(state)
    
    def get_state(self,block_id: BlockId) -> BlockState:
        if block_id < len(self._states):
            return self._state[block_id]
        else: raise RuntimeError(f"BlockID {block_id} does not exist in palette")
        
    def freeze(self) -> None:
        self._frozen = True