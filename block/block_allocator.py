from abc import ABC,abstractmethod
from typing import List,Optional,TypeVar,Protocol,Iterable,Deque,Union
from collections import deque
BlockId=int
class BlockLocation:
    CPU="CPU"
    GPU="GPU"
class Block(ABC):
    @abstractmethod
    def append_token_ids(self,token_ids:List[int])->None:
        pass
    
    @abstractmethod
    def get_block_id(self)->Optional[BlockId]:
        pass
    
    @abstractmethod
    def set_block_id(self,block_id:BlockId)->None:
        self._block_id=block_id
    
    @abstractmethod
    def token_ids(self)->List[int]:
        pass 
    
    @property
    @abstractmethod
    def num_empty_slots(self) -> int:
        pass

    @property
    @abstractmethod
    def is_full(self) -> bool:
        pass
    class Factory(Protocol):
        @abstractmethod
        def __call__(
            self,
            prev_block: Optional["Block"],
            token_ids: List[int],
            block_size: int,
            allocator: "BlockAllocator",
            block_id: Optional[int] = None,
            computed: bool = False,
            extra_hash: Optional[int] = None,
        ) -> "Block":
            pass

class BlockList:
    """This class is an optimization to allow fast-access to physical 
    block ids. It maintains a block id list that is updated with the 
    block list and this avoids the need to reconstruct the block id 
    list on every iteration of the block manager
    """

    def __init__(self, blocks: List[Block]):
        self._blocks: List[Block] = []
        self._block_ids: List[int] = []

        self.update(blocks)

    def _add_block_id(self, block_id: Optional[BlockId]) -> None:
        assert block_id is not None
        self._block_ids.append(block_id)

    def _update_block_id(self, block_index: int,
                         new_block_id: Optional[BlockId]) -> None:
        assert new_block_id is not None
        self._block_ids[block_index] = new_block_id

    def update(self, blocks: List[Block]):
        self._blocks = blocks

        # Cache block ids for fast query
        self._block_ids = []
        for block in self._blocks:
            self._add_block_id(block.block_id)

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

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, block_index: int) -> Block:
        return self._blocks[block_index]

    def __setitem__(self, block_index: int, new_block: Block) -> None:
        self._blocks[block_index] = new_block
        self._update_block_id(block_index, new_block.block_id)

    def reset(self):
        self._blocks = []
        self._block_ids = []

    def list(self) -> List[Block]:
        return self._blocks

    def ids(self) -> List[int]:
        return self._block_ids



class BlockAllocator(ABC):
    @abstractmethod
    def allocate_mutable_block(self, prev_block: Optional[Block],
                               extra_hash: Optional[int]) -> Block:
        pass

    @abstractmethod
    def allocate_immutable_block(self, prev_block: Optional[Block],
                                 token_ids: List[int],
                                 extra_hash: Optional[int]) -> Block:
        pass

    @abstractmethod
    def allocate_immutable_blocks(self, prev_block: Optional[Block],
                                  block_token_ids: List[List[int]],
                                  extra_hash: Optional[int]) -> List[Block]:
        pass

    @abstractmethod
    def free(self, block: Block) -> None:
        pass

    @abstractmethod
    def fork(self, last_block: Block) -> List[Block]:
        pass

    @abstractmethod
    def get_num_total_blocks(self) -> int:
        pass

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        pass

    @abstractmethod
    def get_physical_block_id(self, absolute_id: int) -> int:
        pass

    @abstractmethod
    def swap_out(self, blocks: List[Block]) -> None:
        pass

    @abstractmethod
    def swap_in(self, blocks: List[Block]) -> None:
        pass
    

class BlockPool:
    """Used to pre-allocate block objects, in order to avoid excessive python
    object allocations/deallocations.
    The pool starts from "pool_size" objects and will increase to more objects
    if necessary

    Note that multiple block objects may point to the same physical block id,
    which is why this pool is needed, so that it will be easier to support
    prefix caching and more complicated sharing of physical blocks.
    """

    def __init__(self, block_size: int, create_block: Block.Factory,
                 allocator: BlockAllocator, pool_size: int):
        self._block_size = block_size
        self._create_block = create_block
        self._allocator = allocator
        self._pool_size = pool_size
        assert self._pool_size >= 0

        self._free_ids: Deque[int] = deque(range(self._pool_size))
        self._pool = []
        for i in range(self._pool_size):
            self._pool.append(
                self._create_block(prev_block=None,
                                   token_ids=[],
                                   block_size=self._block_size,
                                   allocator=self._allocator,
                                   block_id=None,
                                   extra_hash=None))

    def increase_pool(self):
        """Doubles the internal pool size
        """
        cur_pool_size = self._pool_size
        new_pool_size = cur_pool_size * 2
        self._pool_size = new_pool_size

        self._free_ids += deque(range(cur_pool_size, new_pool_size))

        for i in range(cur_pool_size, new_pool_size):
            self._pool.append(
                self._create_block(prev_block=None,
                                   token_ids=[],
                                   block_size=self._block_size,
                                   allocator=self._allocator,
                                   block_id=None,
                                   extra_hash=None))

    def init_block(self,
                   prev_block: Optional[Block],
                   token_ids: List[int],
                   block_size: int,
                   physical_block_id: Optional[int],
                   extra_hash: Optional[int] = None) -> Block:
        if len(self._free_ids) == 0:
            self.increase_pool()
            assert len(self._free_ids) > 0

        pool_id = self._free_ids.popleft()

        block = self._pool[pool_id]
        block.__init__(  # type: ignore[misc]
            prev_block=prev_block,
            token_ids=token_ids,
            block_size=block_size,
            allocator=block._allocator,  # type: ignore[attr-defined] 
            block_id=physical_block_id,
            extra_hash=extra_hash)
        block.pool_id = pool_id  # type: ignore[attr-defined]
        return block

    def free_block(self, block: Block) -> None:
        self._free_ids.appendleft(block.pool_id)  # type: ignore[attr-defined]

class NaiveBlockAllocator(BlockAllocator):
    def __init__(
        self,
        create_block: Block.Factory,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
        block_pool: Optional[BlockPool] = None,
    ):
        if block_ids is None:
            block_ids=range(num_blocks)
        self._free_block_indices: Deque[BlockId] = deque(block_ids)
        self._all_block_indices = frozenset(block_ids)
        assert len(self._all_block_indices) == num_blocks

        self._block_size = block_size


        if block_pool is None:
            extra_factor = 4
            # Pre-allocate "num_blocks * extra_factor" block objects.
            # The "* extra_factor" is a buffer to allow more block objects
            # than physical blocks
            self._block_pool = BlockPool(self._block_size, create_block, self,
                                         num_blocks * extra_factor)
        else:
            # In this case, the block pool is provided by the caller,
            # which means that there is most likely a need to share
            # a block pool between allocators
            self._block_pool = block_pool

    def allocate_immutable_block(self,
                                 prev_block: Optional[Block],
                                 token_ids: List[int],
                                 extra_hash: Optional[int] = None,
                                 device: Optional[BlockLocation] = None) -> Block:
        """Allocates a new immutable block with the given token IDs, linked to
        the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.
            token_ids (List[int]): The token IDs to be stored in the new block.

        Returns:
            Block: The newly allocated immutable block.
        """
        assert device is None
        block = self.allocate_mutable_block(prev_block=prev_block)
        block.append_token_ids(token_ids)
        return block

    def allocate_immutable_blocks(
            self,
            prev_block: Optional[Block],
            block_token_ids: List[List[int]],
            extra_hash: Optional[int] = None,
            device: Optional[BlockLocation] = None) -> List[Block]:
        assert device is None
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

    def allocate_mutable_block(self,
                               prev_block: Optional[Block],
                               extra_hash: Optional[int] = None,
                               device: Optional[BlockLocation] = None) -> Block:
        """Allocates a new mutable block, linked to the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.

        Returns:
            Block: The newly allocated mutable block.
        """
        assert device is None
        block_id = self._allocate_block_id()
        block = self._block_pool.init_block(prev_block=prev_block,
                                            token_ids=[],
                                            block_size=self._block_size,
                                            physical_block_id=block_id)
        return block

    def _allocate_block_id(self) -> BlockId:
        if not self._free_block_indices:
            raise BlockAllocator.NoFreeBlocksError()

        block_id = self._free_block_indices.popleft()
        self._refcounter.incr(block_id)
        return block_id

    def _free_block_id(self, block: Union[Block, BlockId]) -> None:
        if isinstance(block, Block):
            block_id = block.block_id
            block.block_id = None
        else:
            block_id = block
        assert block_id is not None

        refcount = self._refcounter.decr(block_id)
        if refcount == 0:
            self._free_block_indices.appendleft(block_id)

    def free(self, block: Block, keep_block_object: bool = False) -> None:
        # Release the physical block id
        self._free_block_id(block)

        # Release the block object
        if not keep_block_object:
            self._block_pool.free_block(block)

    def free_block_id(self, block_id: BlockId) -> None:
        self._free_block_id(block_id)


    def get_num_free_blocks(self) -> int:
        return len(self._free_block_indices)

    def get_num_total_blocks(self) -> int:
        return len(self._all_block_indices)

    def get_physical_block_id(self, absolute_id: int) -> int:
        """Returns the zero-offset block id on certain block allocator
        given the absolute block id.

        Args:
            absolute_id (int): The absolute block id for the block 
            in whole allocator.

        Returns:
            int: The zero-offset block id on certain device.
        """
        return sorted(self._all_block_indices).index(absolute_id)

T = TypeVar("T")
def chunk_list(lst: List[T], chunk_size: int):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]