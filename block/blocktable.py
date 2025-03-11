from typing import Deque,List,Optional,Callable,Iterable
from abc import ABC,abstractmethod
from collections import deque
BlockId=int
TokenId=int
class Block:
    def __init__(
        self,   
        block_size:int,
        block_id:BlockId,
        token_ids:Optional[List[int]]=None
    )->None:
        self._token_ids=[]
        self._token_ids.extend(token_ids)
        self.block_size=block_size
        self.block_id=block_id
    
    def append_tokens(self,token_ids:List[int])->None:
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
    
class BlockPool:
    def __init__(
        self,
        block_size:int,
        pool_size:int
    ):
        self._block_size=block_size
        self._pool_size=pool_size
        self._free_id:Deque[int]=deque(range(pool_size))
        self._pool=[]
        for id in range(pool_size):
            self._pool.append(Block(block_size,id))
    
    def increase_pool(self):
        cur_bool_size=self._pool_size
        self._pool_size=cur_bool_size*2
        self._free_id.extend(range(cur_bool_size,self._pool_size))
        for id in range(cur_bool_size,self._pool_size):
            self._pool.append(Block(self._block_size,id))
    
    def free_block(self,block:Block):
        self._free_id.appendleft(block.block_id)
        
class BlockAllocator:
    def __init__(
        self,
        num_blocks:int,
        block_size:int,
        block_ids: Optional[Iterable[int]] = None,
        block_pool: Optional[BlockPool] = None,
    ):
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
