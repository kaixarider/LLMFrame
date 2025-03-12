from typing import Deque,List,Optional,Callable,Iterable
from abc import ABC,abstractmethod
from enum import Enum
from collections import deque

from MixFrame.util import chunk_list
BlockId=int
TokenId=int
cdiv = lambda a, b: -(a//-b)
class BlockLocation(Enum):
    CPU="cpu"
    GPU="gpu"

class Block:
    def __init__(
        self,   
        prev_block:Optional['Block'],
        block_size:int,
        block_id:BlockId,
        token_ids:Optional[List[int]]=None,
        block_location:BlockLocation=BlockLocation.GPU
    )->None:
        self._token_ids=[]
        self._token_ids.extend(token_ids)
        self.block_size=block_size
        self.block_id=block_id #physical id
        self.pool_id=None #logical id
        self.prev_block=prev_block
        self.block_location=block_location
        
    def append_token_ids(self,token_ids:List[int])->None:
        token_num=len(token_ids)
        if token_num==0:
            return
        assert token_num<self.empty_slots,f"Error!There is no enough slots in \
            block {self.block_id}"
        self._token_ids.extend(token_ids)
    
    def clear_block(self)->None:
        self._token_ids=[]
    
    
    @property
    def empty_slots(self)->int:
        return self.block_size-len(self._token_ids)
    
    def set_blockid(self,value:int)->None:
        self.block_id=value
    
    def set_location(self,location:BlockLocation)->None:
        self.block_location=location

class BlockPool:
    #manage logical blocks
    def __init__(
        self,
        block_size:int,
        pool_size:int,
        pool_location:BlockLocation
    )->None:
        self._block_size=block_size
        self._pool_size=pool_size
        self._free_id:Deque[int]=deque(range(pool_size))
        self.pool_location=pool_location
        self._pool:List[Block]=[]
        for id in range(pool_size):
            self._pool.append(Block(prev_block=None,block_size=block_size,
                                    block_id=id,token_ids=None,
                                    block_location=pool_location))
    
    def increase_pool(self)->None:
        cur_bool_size=self._pool_size
        self._pool_size=cur_bool_size*2
        self._free_id.extend(range(cur_bool_size,self._pool_size))
        for id in range(cur_bool_size,self._pool_size):
            self._pool.append(Block(self._block_size,id))
    
    def free_block(self,block:Block)->None:
        self._free_id.appendleft(block.block_id)
    
    def init_block(self,
                   token_id:List[int],
                   prev_block:Block,
                   block_size:int,
                   physical_block_id:Optional[int],
                   )->Block:
        if len(self._free_id) == 0:
            self.increase_pool()
            assert len(self._free_id) > 0

        pool_id = self._free_id.popleft()
        block = self._pool[pool_id]
        block.__init__(
            block_size=block_size,
            prev_block=prev_block,
            block_id=physical_block_id,
            token_id=token_id,
            block_location=self.pool_location
        )
        block.pool_id=pool_id

class BlockList:
    def __init__(
        self,
        blocks:List[Block]
    ):
        self._blocks:List[Block]=[]
        self._block_id:List[int]=[]
        self.update(blocks)
        
    def update(self,blocks:List[Block]):
        self._blocks=blocks
        self._block_id=[]
        for block in blocks:
            self._block_id.append(block.block_id)
    
    def _add_block_id(self,block_id:BlockId)->None:
        self._block_id.append(block_id)
    
    def _update_block_id(self, block_index: int,
                         new_block_id: Optional[BlockId]) -> None:
        assert new_block_id is not None
        self._block_id[block_index] = new_block_id
        
    def append_token_ids(self, block_index: int, token_ids: List[int]) -> None:
        block = self._blocks[block_index]
        prev_block_id = block.block_id

        block.append_token_ids(token_ids)

        # CoW or promotion may update the internal block_id
        if prev_block_id != block.block_id:
            self._update_block_id(block_index, block.block_id)
    
    def append(self, new_block: Block):
        self._blocks.append(new_block)
        self._add_block_id(new_block.block_id)
    def reset(self):
        self._blocks = []
        self._block_id = []

    def list(self) -> List[Block]:
        return self._blocks

    def ids(self) -> List[int]:
        return self._block_id

class BlockAllocator:
    def __init__(
        self,
        num_blocks:int,
        block_size:int,
        block_ids: Optional[Iterable[int]] = None,
        block_pool: Optional[BlockPool] = None,
    )->None:
        if block_ids is None:
            block_ids = range(num_blocks)
        self._free_block_indices:Deque[int]=deque(block_ids)
        self._all_block_indices = frozenset(block_ids)
        assert len(self._all_block_indices) == num_blocks
        self._block_size = block_size
        if block_pool is None:
            # Pre-allocate "num_blocks * extra_factor" block objects.
            # The "* extra_factor" is a buffer to allow more block objects
            # than physical blocks
            extra_factor=4
            self._block_pool = BlockPool(self._block_size, 
                                         num_blocks * extra_factor)
        else:
            # In this case, the block pool is provided by the caller,
            # which means that there is most likely a need to share
            # a block pool between allocators
            self._block_pool = block_pool
    def _allocate_block_id(self)->int:
        block_id = self._free_block_indices.popleft()
        return block_id
    
    def allocate_mutable_block(
        self,
        prev_block: Optional[Block],
        location:BlockLocation
    )->Block:
        block_id = self._allocate_block_id()
        block=self._block_pool.init_block(
            prev_block=prev_block,
            token_id=[],
            prev_block=prev_block,
            block_size=self._block_size,
            physical_block_id=block_id,  
        )
        return block
    
    def allocate_immutable_block(self,
                                 prev_block: Optional[Block],
                                 token_ids: List[int],
                                 location: Optional[BlockLocation] = None) -> Block:

        block = self.allocate_mutable_block(prev_block=prev_block,location=location)
        block.append_token_ids(token_ids)
        return block
    
    def allocate_immutable_blocks(
            self,
            prev_block: Optional[Block],
            block_token_ids: List[List[int]],
            device: Optional[BlockLocation]=BlockLocation.GPU) -> List[Block]:
        num_blocks = len(block_token_ids)

        block_ids = []
        for i in range(num_blocks):
            block_ids.append(self._allocate_block_id())

        blocks = []
        for i in range(num_blocks):
            prev_block = self._block_pool.init_block(
                prev_block=prev_block,
                token_ids=block_token_ids[i],
                block_size=self._block_size,
                physical_block_id=block_ids[i])
            blocks.append(prev_block)
        return blocks
    
class BlockTable:
    def __init__(
        self,
        block_size:int,
        block_allocator:BlockAllocator,
        blocks:Optional[List[Block]]=None,
        max_block_sliding_window: Optional[int] = None
    ):
        self._block_size=block_size
        self._allocator=block_allocator
        if blocks is None:
            blocks=[]
        self._blocks: BlockList = BlockList(blocks)
        self._max_block_sliding_window = max_block_sliding_window
        
    @staticmethod
    def get_num_required_blocks(token_ids:List[int],
                                block_size:int,
                                ahead_slots:int=0):
        return cdiv (len(token_ids) + ahead_slots, block_size)
    
    def allocate_blocks_for_token_ids(
        self,
        token_ids:List[int],
        block_size:int,
        location:BlockLocation=BlockLocation.GPU,
        ahead_slots:int=0
    )->List[Block]:
        blocks:List[Block]=[]
        block_token_ids = []
        tail_token_ids = []
        for cur_token_ids in chunk_list(token_ids,block_size):
            if len(cur_token_ids) == self._block_size:
                block_token_ids.append(cur_token_ids)
            else:
                tail_token_ids.append(cur_token_ids)
        if block_token_ids:
            blocks.extend(
                self._allocator.allocate_immutable_blocks(prev_block,location))
            prev_block = blocks[-1]
        if tail_token_ids:
            assert len(tail_token_ids) == 1
            cur_token_ids = tail_token_ids[0]

            block = self._allocator.allocate_mutable_block(
                prev_block=prev_block,)
            block.append_token_ids(cur_token_ids)

            blocks.append(block)

        return blocks
    
    def allocate(self,
                 token_ids:List[int],
                 location:BlockLocation=BlockLocation.GPU)->None:
        blocks = self.allocate_blocks_for_token_ids(prev_block=None,
                                                     token_ids=token_ids,block_size=self._block_size
                                                )
        self.update(blocks)
        self._num_full_slots = len(token_ids)
        
    def update(self, blocks: List[Block]) -> None:
        """Resets the table to the newly provided blocks 
        (with their corresponding block ids)
        """
        self._blocks.update(blocks)
    
    
           
        