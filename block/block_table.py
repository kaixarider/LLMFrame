from typing import List,Optional
from block_allocator import Block,BlockAllocator,BlockList,BlockLocation,chunk_list
class BlockTable:
    def __init__(
        self,
        block_size:int,
        block_allocator:BlockAllocator,
        _blocks:Optional[List[Block]]=None,
        max_block_sliding_window: Optional[int] = None
    ):
        self._block_size=block_size
        self._allocator=block_allocator
        if _blocks is None:
            _blocks = []
        self._blocks: BlockList = BlockList(_blocks)
        
    def get_num_required_blocks(self,token_ids:List[int],ahead_num:int=0)->int:
        '''get the number of required blocks'''
        return -((len(token_ids)+ahead_num)/-self._block_size)
    
    def allocate(self,
                 token_ids:List[int],
                 location:BlockLocation,
                 extra_hash:Optional[int]=None)->None:
        assert not self._is_allocated
        assert token_ids
        blocks = self._allocate_blocks_for_token_ids(prev_block=None,
                                                     token_ids=token_ids,
                                                     device=location,
                                                     extra_hash=extra_hash)
        self.update(blocks)
        self._num_full_slots = len(token_ids)
    
    def _allocate_blocks_for_token_ids(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        device: BlockLocation,
        extra_hash: Optional[int] = None) -> List[Block]:
        blocks: List[Block] = []

        block_token_ids = []
        tail_token_ids = []
        for cur_token_ids in chunk_list(token_ids, self._block_size):
            if len(cur_token_ids) == self._block_size:
                block_token_ids.append(cur_token_ids)
            else:
                tail_token_ids.append(cur_token_ids)

        if block_token_ids:
            blocks.extend(
                self._allocator.allocate_immutable_blocks(
                    prev_block,
                    block_token_ids=block_token_ids,
                    device=device,
                    extra_hash=extra_hash))
            prev_block = blocks[-1]

        if tail_token_ids:
            assert len(tail_token_ids) == 1
            cur_token_ids = tail_token_ids[0]

            block = self._allocator.allocate_mutable_block(
                prev_block=prev_block, device=device, extra_hash=extra_hash)
            block.append_token_ids(cur_token_ids)

            blocks.append(block)

        return blocks
    

    def update(self, blocks: List[Block]) -> None:
        """Resets the table to the newly provided blocks 
        (with their corresponding block ids)
        """
        self._blocks.update(blocks)