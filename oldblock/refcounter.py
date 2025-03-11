from typing import Protocol,Iterable,Dict,List,Tuple,Optional
from block_allocator import Block
BlockId=int
RefCount=int
class RefCounterProtocol(Protocol):

    def incr(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError

    def decr(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError

    def get(self, block_id: BlockId) -> RefCount:
        raise NotImplementedError
class RefCounter(RefCounterProtocol):
    def __init__(self,all_blocks:Iterable[BlockId])->None:
        dedup=set(all_blocks)
        self._refcount=Dict[BlockId,RefCount]={
            idx:0
            for idx in dedup
        }
    
    def incr(self,block_id:BlockId)->RefCount:
        assert block_id in self._refcount,f'Error!block id {block_id} doesn\'t exist'
        ref_num=self._refcount[block_id]
        ref_num+=1
        self._refcount=ref_num
        return ref_num
    
    def decr(self,block_id:BlockId)->RefCount:
        assert block_id in self._refcount,f'Error!block id {block_id} doesn\'t exist'
        ref_num=self._refcount[block_id]
        ref_num-=1
        assert ref_num>=0,f'Error!ref_count of block {block_id} shouldn\'t be less than 0'
        return ref_num
    
    def get(self,block_id:BlockId)->RefCount:
        assert block_id in self._refcount,f'Error!block id {block_id} doesn\'t exist'
        return self._refcount[block_id]
    
    def as_readonly(self)->"ReadOnlyRefCounter":
        return 

class ReadOnlyRefCounter(RefCounterProtocol):
    def __init__(self, refcounter: RefCounter):
        self._refcounter = refcounter
        
    def incr(self, block_id: BlockId) -> RefCount:
        raise ValueError("Read Only!Modifying is not allowed!")

    def decr(self, block_id: BlockId) -> RefCount:
        raise ValueError("Read Only!Modifying is not allowed!")

    def get(self,block_id:BlockId)->RefCount:
        return self._refcounter.get(block_id)

class CopyOnWriteTracker:
    """A class for tracking and managing copy-on-write operations for blocks.

    The CopyOnWriteTracker class maintains a mapping of source block indices to
        their corresponding copy-on-write destination block indices. It works in
        conjunction with a RefCounter.

    Args:
        refcounter (RefCounter): The reference counter used to track block
            reference counts.
    """

    def __init__(self, refcounter: RefCounterProtocol):
        self._copy_on_writes: List[Tuple[BlockId, BlockId]] = []
        self._refcounter = refcounter

    def is_appendable(self, block: Block) -> bool:
        """Checks if the block is shared or not. If shared, then it cannot
        be appended and needs to be duplicated via copy-on-write
        """
        block_id = block.block_id
        if block_id is None:
            return True

        refcount = self._refcounter.get(block_id)
        return refcount <= 1

    def record_cow(self, src_block_id: Optional[BlockId],
                   trg_block_id: Optional[BlockId]) -> None:
        """Records a copy-on-write operation from source to target block id
        Args:
            src_block_id (BlockId): The source block id from which to copy 
                the data
            trg_block_id (BlockId): The target block id to which the data
                is copied
        """
        assert src_block_id is not None
        assert trg_block_id is not None
        self._copy_on_writes.append((src_block_id, trg_block_id))

    def clear_cows(self) -> List[Tuple[BlockId, BlockId]]:
        """Clears the copy-on-write tracking information and returns the current
        state.

        This method returns a list mapping source block indices to
         destination block indices for the current copy-on-write operations.
        It then clears the internal tracking information.

        Returns:
            List[Tuple[BlockId, BlockId]]: A list mapping source
                block indices to destination block indices for the
                current copy-on-write operations.
        """
        cows = self._copy_on_writes
        self._copy_on_writes = []
        return cows
