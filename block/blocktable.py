from typing import Deque,List,Optional,Callable,Iterable,Union
from abc import ABC,abstractmethod
from enum import Enum
from collections import deque
import math

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
        block_location:BlockLocation,
        token_ids:Optional[List[int]]=None,
        
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
            self._pool.append(Block(self._block_size,id,self.pool_location,None))
    
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
        return block
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
        location:BlockLocation,
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
                                         num_blocks * extra_factor,location)
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
    
    def free(self, block: Block, keep_block_object: bool = False) -> None:
        # Release the physical block id
        self._free_block_id(block)

        # Release the block object
        if not keep_block_object:
            self._block_pool.free_block(block)
        
    def _free_block_id(self, block: Union[Block, BlockId]) -> None:
        if isinstance(block, Block):
            block_id = block.block_id
            block.block_id = None
        else:
            block_id = block
        assert block_id is not None
        self._free_block_indices.appendleft(block_id)
    
    def get_num_free_block(self):
        return len(self._free_block_indices)
    

class BlockTable:
    '''record blocks a req use'''
    '''缺少migrate后取出blocks的函数'''
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
        
        self._num_full_slots=self._get_num_token_ids()
    @staticmethod
    def get_num_required_blocks(token_ids:List[int],
                                block_size:int,
                                ahead_slots:int=0):
        return cdiv (len(token_ids) + ahead_slots, block_size)
    def _get_num_token_ids(self) -> int:
        res = 0
        for block in self.blocks:
            res += len(block._token_ids)
        return res
    def _copy_blocks(
        self,
        other_token_blocks:List[List[int]]
    )->List[Block]:
        blocks:List[Block]=[]
        block_token_ids:List[int] = []
        tail_token_ids:List[int] = []
        for tokens in other_token_blocks:
            if len(tokens)==self._block_size:
                block_token_ids.append(tokens)
            else:
                tail_token_ids.append(tokens)
        if block_token_ids:
            blocks.extend(
                self._allocator.allocate_immutable_blocks(prev_block,BlockLocation.GPU))
            prev_block = blocks[-1]
        if tail_token_ids:
            assert len(tail_token_ids) == 1
            cur_token_ids = tail_token_ids[0]

            block = self._allocator.allocate_mutable_block(
                prev_block=prev_block,)
            block.append_token_ids(cur_token_ids)

            blocks.append(block)

        return blocks
    def _allocate_blocks_for_token_ids(
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
        blocks = self._allocate_blocks_for_token_ids(prev_block=None,
                                                     token_ids=token_ids,block_size=self._block_size
                                                )
        self.update(blocks)
        self._num_full_slots = len(token_ids)
        
    def update(self, blocks: List[Block]) -> None:
        """Resets the table to the newly provided blocks 
        (with their corresponding block ids)
        """
        self._blocks.update(blocks)
    def used_blocks(self)->List[Block]:
        return self._blocks._blocks
    def free(self)->None:
        for block in self.blocks:
            self._allocator.free(block)
        self._blocks.reset()

    @property
    def blocks(self)->List[Block]:
        return self._blocks.list()
    @property
    def physical_block_ids(self) -> List[int]:
        """Returns a list of physical block indices for the blocks in the
        BlockTable.

        This property returns a list of integers, where each integer represents
        the physical block index of a corresponding block in the `_blocks` list.
        The physical block index is a unique identifier for the memory location
        occupied by the block.

        Returns:
            List[int]: A list of physical block indices for the blocks in the
                BlockTable.
        """
        return self._blocks.ids()
    @property
    def _num_empty_slots(self) -> int:
        return len(self._blocks) * self._block_size - self._num_full_slots
    
    def get_num_blocks_touched_by_append_slots(
            self, token_ids: List[int], num_lookahead_slots: int) -> int:
        """Determine how many blocks will be "touched" by appending the token
        ids.

        This is required for the scheduler to determine whether a sequence can
        continue generation, or if it must be preempted.
        """
        # Math below is equivalent to:
        # all_token_ids = token_ids + [-1] * num_lookahead_slots
        # token_blocks = self._chunk_token_blocks_for_append(all_token_ids)
        # return len(token_blocks)

        num_token_ids = len(token_ids) + num_lookahead_slots
        first_chunk_size = self._block_size - (self._num_full_slots %
                                               self._block_size)
        num_token_blocks = (1 + math.ceil(
            (num_token_ids - first_chunk_size) / self._block_size))
        return num_token_blocks
    
    def get_unseen_token_ids(self, req_token_ids: List[int]) -> List[int]:
        """Get the number of "unseen" tokens in the sequence.

        Unseen tokens are tokens in the sequence corresponding to this block
        table, but are not yet appended to this block table.

        Args:
            sequence_token_ids (List[int]): The list of token ids in the
                sequence.

        Returns:
            List[int]: The postfix of sequence_token_ids that has not yet been
                appended to the block table.
        """

        # Since the block table is append-only, the unseen token ids are the
        # ones after the appended ones.
        return req_token_ids[self._num_full_slots:]
           
    def append_token_ids(self,
                         token_ids: List[int],
                         ahead_slots: int = 0,
                         num_computed_slots: Optional[int] = None) -> None:
        """Appends a sequence of token IDs to the existing blocks in the
        BlockTable.

        This method appends the given sequence of token IDs to the existing
        blocks in the BlockTable. If there is not enough space in the existing
        blocks, new blocks are allocated using the `ensure_num_empty_slots`
        method to accommodate the additional tokens.

        The token IDs are divided into chunks of size `block_size` (except for
        the first chunk, which may be smaller), and each chunk is appended to a
        separate block.

        Args:
            token_ids (List[int]): The sequence of token IDs to be appended.
            num_computed_slots (Optional[int]): The number of KV cache slots
                that are already filled (computed).
                When sliding window is enabled, this is used to compute how many
                blocks to drop at the front of the sequence.
                Without sliding window, None can be passed.
                Without chunked prefill, it should be the same as
                _num_full_slots.
            extra_hash (Optional[int]): The hash value of additional
                factors such as adapters that influence the block, apart
                from the token_ids.
        """
        assert len(self._blocks) > 0

        # Drop blocks that are no longer needed due to sliding window
        if self._max_block_sliding_window is not None:
            null_block = self._allocator.allocate_or_get_null_block()
            assert num_computed_slots is not None
            end_block_idx = (num_computed_slots //
                             self._block_size) - self._max_block_sliding_window
            for idx in range(0, end_block_idx):
                b = self._blocks[idx]
                if b is not null_block:
                    self._allocator.free(b)
                    self._blocks[idx] = null_block

        # Ensure there are enough empty slots for the new tokens plus
        # lookahead slots
        self.ensure_num_empty_slots(num_empty_slots=len(token_ids) +
                                    ahead_slots)

        # Update the blocks with the new tokens
        first_block_idx = self._num_full_slots // self._block_size
        token_blocks = self._chunk_token_blocks_for_append(token_ids)

        for i, token_block in enumerate(token_blocks):
            self._blocks.append_token_ids(first_block_idx + i, token_block)

        self._num_full_slots += len(token_ids)
    
    def ensure_num_empty_slots(self,
                               num_empty_slots: int) -> None:
        """Ensures that the BlockTable has at least the specified number of
        empty slots available.

        This method checks if the BlockTable has enough empty slots (i.e.,
        available space) to accommodate the requested number of tokens. If not,
        it allocates additional blocks on the GPU to ensure that the required
        number of empty slots is available.

        Args:
            num_empty_slots (int): The minimum number of empty slots required.
            extra_hash (Optional[int]): The hash value of additional
                factors such as adapters that influence the block, apart
                from the token_ids.
        """
        # Currently the block table only supports
        # appending tokens to GPU blocks.

        if self._num_empty_slots >= num_empty_slots:
            return

        slots_to_allocate = num_empty_slots - self._num_empty_slots
        blocks_to_allocate = cdiv(slots_to_allocate, self._block_size)

        for _ in range(blocks_to_allocate):
            assert len(self._blocks) > 0
            self._blocks.append(
                self._allocator.allocate_mutable_block(
                    prev_block=self._blocks[-1],
                    location=BlockLocation.GPU))
    def _chunk_token_blocks_for_append(
            self, token_ids: List[int]) -> List[List[int]]:
        """Split the token ids into block-sized chunks so they can be easily
        appended to blocks. The first such "token block" may have less token ids
        than the block size, since the last allocated block may be partially
        full.

        If no token ids are provided, then no chunks are returned.
        """

        if not token_ids:
            return []

        first_chunk_size = self._block_size - (self._num_full_slots %
                                               self._block_size)
        token_blocks = [token_ids[:first_chunk_size]]
        token_blocks.extend(
            chunk_list(token_ids[first_chunk_size:], self._block_size))
        return token_blocks