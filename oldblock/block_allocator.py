from abc import ABC,abstractmethod
from typing import List,Optional,TypeVar,Protocol,Iterable,Deque,Union,Tuple,Dict,FrozenSet
from refcounter import RefCounter,RefCounterProtocol,CopyOnWriteTracker
from collections import deque

from MixFrame.request.request import Request
BlockId=int
'''Only support naive block'''
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
        self._refcounter = RefCounter(
            all_block_indices=self._free_block_indices)
        self._block_size = block_size
        self._cow_tracker = CopyOnWriteTracker(
            refcounter=self._refcounter.as_readonly())

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
    def cow_block_if_not_appendable(self, block: Block) -> BlockId:
        """Performs a copy-on-write operation on the given block if it is not
        appendable.

        Args:
            block (Block): The block to check for copy-on-write.

        Returns:
            BlockId: The block index of the new block if a copy-on-write 
                operation was performed, or the original block index if
                no copy-on-write was necessary.
        """
        src_block_id = block.block_id
        assert src_block_id is not None

        if self._cow_tracker.is_appendable(block):
            return src_block_id

        self._free_block_id(block)
        trg_block_id = self._allocate_block_id()

        self._cow_tracker.record_cow(src_block_id, trg_block_id)

        return trg_block_id

    def clear_copy_on_writes(self) -> List[Tuple[BlockId, BlockId]]:
        """Returns the copy-on-write source->destination mapping and clears it.

        Returns:
            List[Tuple[BlockId, BlockId]]: A list mapping source
                block indices to destination block indices.
        """
        return self._cow_tracker.clear_cows()

    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        """Mark blocks as accessed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        pass

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        """Mark blocks as computed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        pass

    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        """Determine blocks that can be skipped in prefill.

        Since the naive allocator does not support prefix caching, always return
        an empty list.
        """
        return []

    def promote_to_immutable_block(self, block: Block) -> BlockId:
        raise NotImplementedError("There is no promotion for naive blocks")

    def get_num_full_blocks_touched(self, blocks: List[Block]) -> int:
        """Returns the number of full blocks that will be touched by
        swapping in/out.

        Args:
            blocks: List of blocks to be swapped.
        Returns:
            int: the number of full blocks that will be touched by
                swapping in/out the given blocks. Non full blocks are ignored
                when deciding the number of blocks to touch.
        """
        # NOTE: for naive block, we use set to eliminate common blocks among
        # seqs, also we compare the empty slots in the mutable blocks with
        # lookahead slots to get the number of unique new block that are
        # needed.
        old_block_set = set()
        for block in blocks:
            if block.is_full:
                old_block_set.add(block)
        return len(old_block_set)

    def swap_out(self, blocks: List[Block]) -> None:
        for block in blocks:
            self._free_block_id(block)

    def swap_in(self, blocks: List[Block]) -> None:
        for block in blocks:
            # Here we allocate either immutable or mutable block and then
            # extract its block_id. Note that the block object is released
            # and the block_id is assigned to "block" to allow reusing the
            # existing "block" object
            if block.is_full:
                tmp_block = self.allocate_immutable_block(
                    prev_block=block.prev_block, token_ids=block.token_ids)
            else:
                tmp_block = self.allocate_mutable_block(
                    prev_block=block.prev_block)
                tmp_block.append_token_ids(block.token_ids)

            block_id = tmp_block.block_id
            tmp_block.block_id = None
            self._block_pool.free_block(tmp_block)

            block.block_id = block_id  # Assign block_id

    def get_prefix_cache_hit_rate(self) -> float:
        return -1

    def reset_prefix_cache(self) -> bool:
        """No prefix cache for naive block allocator."""
        return True

    def find_cached_blocks_prefix(self, block_hashes: List[int]) -> List[int]:
        # Not applicable for naive block allocator.
        return []

class NaiveBlock(Block):
    """An implementation of the Block class that does not support prefix
    caching.

    The NaiveBlock class represents a block of token IDs with a fixed size. It
    provides methods for appending token IDs to the block and manages copy-on
    -write operations when necessary.

    Args:
        prev_block (Block): The previous block in the sequence.
        token_ids (List[int]): The initial token IDs to be stored in the block.
        block_size (int): The maximum number of token IDs that can be stored in
            the block.
        allocator (BlockAllocator): The block allocator associated with this
            block.
        block_id (Optional[int], optional): The physical block index
            of this block. Defaults to None, which means no allocation has been
            made.
        _cow_target (Optional[Block], optional): The copy-on-write target block.
            If not provided, it defaults to self.
    """

    def __init__(self,
                 prev_block: Optional[Block],
                 token_ids: List[int],
                 block_size: int,
                 allocator: BlockAllocator,
                 block_id: Optional[int] = None,
                 _cow_target: Optional[Block] = None,
                 extra_hash: Optional[int] = None):
        self._token_ids: List[int] = []
        self._block_size = block_size
        self._prev_block = prev_block
        self._block_id = block_id
        self._allocator = allocator
        self._cow_target = _cow_target if _cow_target is not None else self

        self._append_token_ids_no_cow(token_ids)

    def append_token_ids(self, token_ids: List[int]) -> None:
        """Appends the given token IDs to the block and performs a 
        copy-on-write if necessary.

        Args:
            token_ids (Optional[List[int]]): The token IDs to be appended 
                to the block.
        """
        self._append_token_ids_no_cow(token_ids)

        if self._block_id is not None:
            self._block_id = (self._allocator.cow_block_if_not_appendable(
                self._cow_target))

    def _append_token_ids_no_cow(self, token_ids: List[int]) -> None:
        """Appends the given token IDs to the block

        Args:
            token_ids (List[int]): The token IDs to be appended to the block.
        """
        if len(token_ids) == 0:
            return

        assert len(token_ids) <= self.num_empty_slots

        self._token_ids.extend(token_ids)

    @property
    def computed(self) -> bool:
        raise NotImplementedError

    @computed.setter
    def computed(self, value) -> None:
        raise NotImplementedError

    @property
    def last_accessed(self) -> float:
        raise NotImplementedError

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float):
        raise NotImplementedError

    @property
    def block_id(self) -> Optional[int]:
        return self._block_id

    @block_id.setter
    def block_id(self, value: Optional[int]) -> None:
        self._block_id = value

    @property
    def is_full(self) -> bool:
        return self.num_empty_slots == 0

    @property
    def num_empty_slots(self) -> int:
        return self._block_size - len(self.token_ids)

    @property
    def token_ids(self) -> List[int]:
        return self._token_ids

    @property
    def num_tokens_total(self) -> int:
        raise NotImplementedError(
            "num_tokens_total is not used for naive block")

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def prev_block(self) -> Optional["Block"]:
        return self._prev_block

    @property
    def extra_hash(self):
        return None

    @property
    def content_hash(self) -> Optional[int]:
        return None

class DeviceAwareBlockAllocator(ABC):

    @abstractmethod
    def allocate_mutable_block(self,
                               prev_block: Optional[Block],
                               device: BlockLocation,
                               extra_hash: Optional[int] = None) -> Block:
        pass

    @abstractmethod
    def allocate_immutable_block(self,
                                 prev_block: Optional[Block],
                                 token_ids: List[int],
                                 device: BlockLocation,
                                 extra_hash: Optional[int] = None) -> Block:
        pass

    @abstractmethod
    def allocate_immutable_blocks(
        self,
        prev_block: Optional[Block],
        block_token_ids: List[List[int]],
        device: BlockLocation,
        extra_hash: Optional[int] = None,
    ) -> List[Block]:
        pass

    @abstractmethod
    def get_num_free_blocks(self, device: BlockLocation) -> int:
        pass

    @abstractmethod
    def get_num_total_blocks(self, device: BlockLocation) -> int:
        pass

    @abstractmethod
    def free(self, block: Block) -> None:
        pass

    @abstractmethod
    def fork(self, last_block: Block) -> List[Block]:
        pass

    @property
    @abstractmethod
    def all_block_ids(self) -> FrozenSet[int]:
        pass

    @abstractmethod
    def clear_copy_on_writes(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        pass

    @abstractmethod
    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        pass

    @abstractmethod
    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        pass

    @abstractmethod
    def get_num_full_blocks_touched(self, blocks: List[Block],
                                    device: BlockLocation) -> int:
        pass

    @abstractmethod
    def swap(self, blocks: List[Block], src_device: BlockLocation,
             dst_device: BlockLocation) -> Dict[int, int]:
        pass

    @abstractmethod
    def get_physical_block_id(self, device: BlockLocation, absolute_id: int) -> int:
        pass

    @abstractmethod
    def allocate_or_get_null_block(self) -> Block:
        """
        Null blocks are used as a placeholders for KV cache blocks that have
        been dropped due to sliding window.
        There is at most one null block per allocator.
        """
        pass

    @abstractmethod
    def get_prefix_cache_hit_rate(self, device: BlockLocation) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        pass

    @abstractmethod
    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache."""
        pass

    @abstractmethod
    def find_cached_blocks_prefix(
        self,
        block_hashes: List[int],
        device: BlockLocation = BlockLocation.GPU,
    ) -> List[int]:
        pass

class CpuGpuBlockAllocator(DeviceAwareBlockAllocator):
    """A block allocator that can allocate blocks on both CPU and GPU memory.

    This class implements the `DeviceAwareBlockAllocator` interface and provides
    functionality for allocating and managing blocks of memory on both CPU and
    GPU devices.

    The `CpuGpuBlockAllocator` maintains separate memory pools for CPU and GPU
    blocks, and allows for allocation, deallocation, forking, and swapping of
    blocks across these memory pools.
    """

    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
    ) -> DeviceAwareBlockAllocator:
        """Creates a CpuGpuBlockAllocator instance with the specified
        configuration.

        This static method creates and returns a CpuGpuBlockAllocator instance
        based on the provided parameters. It initializes the CPU and GPU block
        allocators with the specified number of blocks, block size, and
        allocator type.

        Args:
            allocator_type (str): The type of block allocator to use for CPU
                and GPU blocks. Currently supported values are "naive" and
                "prefix_caching".
            num_gpu_blocks (int): The number of blocks to allocate for GPU
                memory.
            num_cpu_blocks (int): The number of blocks to allocate for CPU
                memory.
            block_size (int): The size of each block in number of tokens.

        Returns:
            DeviceAwareBlockAllocator: A CpuGpuBlockAllocator instance with the
                specified configuration.

        Notes:
            - The block IDs are assigned contiguously, with GPU block IDs coming
                before CPU block IDs.
        """
        # For HPU, block id 0 is used only for padding
        reserved_blocks = 0
        block_ids = list(
            range(reserved_blocks, num_gpu_blocks + num_cpu_blocks))
        num_gpu_blocks -= reserved_blocks
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]

        if allocator_type == "naive":
            gpu_allocator: BlockAllocator = NaiveBlockAllocator(
                create_block=NaiveBlock,  # type: ignore
                num_blocks=num_gpu_blocks,
                block_size=block_size,
                block_ids=gpu_block_ids,
            )

            cpu_allocator: BlockAllocator = NaiveBlockAllocator(
                create_block=NaiveBlock,  # type: ignore
                num_blocks=num_cpu_blocks,
                block_size=block_size,
                block_ids=cpu_block_ids,
            )
        else:
            raise ValueError(f"Unknown allocator type {allocator_type=}")

        return CpuGpuBlockAllocator(
            cpu_block_allocator=cpu_allocator,
            gpu_block_allocator=gpu_allocator,
        )

    def __init__(self, cpu_block_allocator: BlockAllocator,
                 gpu_block_allocator: BlockAllocator):
        assert not (
            cpu_block_allocator.all_block_ids
            & gpu_block_allocator.all_block_ids
        ), "cpu and gpu block allocators can't have intersection of block ids"

        self._allocators = {
            BlockLocation.CPU: cpu_block_allocator,
            BlockLocation.GPU: gpu_block_allocator,
        }

        self._swap_mapping: Dict[int, int] = {}
        self._null_block: Optional[Block] = None

        self._block_ids_to_allocator: Dict[int, BlockAllocator] = {}
        for _, allocator in self._allocators.items():
            for block_id in allocator.all_block_ids:
                self._block_ids_to_allocator[block_id] = allocator

    def allocate_mutable_block(self,
                               prev_block: Optional[Block],
                               device: BlockLocation,
                               extra_hash: Optional[int] = None) -> Block:
        """Allocates a new mutable block on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block to in the sequence.
                Used for prefix hashing.
            device (BlockLocation): The device on which to allocate the new block.
            extra_hash (Optional[int]): The hash value of additional
                factors, such as adapters, that influence the block hash
                in the prefix caching block.

        Returns:
            Block: The newly allocated mutable block.
        """
        return self._allocators[device].allocate_mutable_block(
            prev_block, extra_hash=extra_hash)

    def allocate_immutable_blocks(
            self,
            prev_block: Optional[Block],
            block_token_ids: List[List[int]],
            device: BlockLocation,
            extra_hash: Optional[int] = None) -> List[Block]:
        """Allocates a new group of immutable blocks with the provided block 
        token IDs on the specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            block_token_ids (List[int]): The list of block token IDs to be 
                stored in the new blocks.
            device (BlockLocation): The device on which to allocate the new block.
            extra_hash (Optional[int]): The hash value of additional
                factors, such as adapters, that influence the block hash
                in the prefix caching block.

        Returns:
            List[Block]: The newly allocated list of immutable blocks 
                containing the provided block token IDs.
        """
        return self._allocators[device].allocate_immutable_blocks(
            prev_block, block_token_ids, extra_hash=extra_hash)

    def allocate_immutable_block(self,
                                 prev_block: Optional[Block],
                                 token_ids: List[int],
                                 device: BlockLocation,
                                 extra_hash: Optional[int] = None) -> Block:
        """Allocates a new immutable block with the provided token IDs on the
        specified device.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence.
                Used for prefix hashing.
            token_ids (List[int]): The list of token IDs to be stored in the new
                block.
            device (BlockLocation): The device on which to allocate the new block.
            extra_hash (Optional[int]): The hash value of additional
                factors, such as adapters, that influence the block hash
                in the prefix caching block.

        Returns:
            Block: The newly allocated immutable block containing the provided
                token IDs.
        """
        return self._allocators[device].allocate_immutable_block(
            prev_block, token_ids, extra_hash=extra_hash)

    def free(self, block: Block) -> None:
        """Frees the memory occupied by the given block.

        Args:
            block (Block): The block to be freed.
        """
        block_id = block.block_id
        assert block_id is not None
        allocator = self._block_ids_to_allocator[block_id]
        allocator.free(block)

    def fork(self, last_block: Block) -> List[Block]:
        """Creates a new sequence of blocks that shares the same underlying
            memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: A new list of blocks that shares the same memory as the
                original sequence.
        """
        # do not attempt to fork the null block
        block_id = last_block.block_id
        assert block_id is not None
        allocator = self._block_ids_to_allocator[block_id]
        return allocator.fork(last_block)

    def get_num_free_blocks(self, device: BlockLocation) -> int:
        """Returns the number of free blocks available on the specified device.

        Args:
            device (BlockLocation): The device for which to query the number of free
                blocks. AssertionError is raised if None is passed.

        Returns:
            int: The number of free blocks available on the specified device.
        """
        return self._allocators[device].get_num_free_blocks()

    def get_num_total_blocks(self, device: BlockLocation) -> int:
        return self._allocators[device].get_num_total_blocks()

    def get_physical_block_id(self, device: BlockLocation, absolute_id: int) -> int:
        """Returns the zero-offset block id on certain device given the 
        absolute block id.

        Args:
            device (BlockLocation): The device for which to query relative block id.
                absolute_id (int): The absolute block id for the block in 
                whole allocator.

        Returns:
            int: The zero-offset block id on certain device.
        """
        return self._allocators[device].get_physical_block_id(absolute_id)

    def swap(self, blocks: List[Block], src_device: BlockLocation,
             dst_device: BlockLocation) -> Dict[int, int]:
        """Execute the swap for the given blocks from source_device
        on to dest_device, save the current swap mapping and append 
        them to the accumulated `self._swap_mapping` for each 
        scheduling move.

        Args:
            blocks: List of blocks to be swapped.
            src_device (BlockLocation): BlockLocation to swap the 'blocks' from.
            dst_device (BlockLocation): BlockLocation to swap the 'blocks' to.
        
        Returns:
            Dict[int, int]: Swap mapping from source_device
                on to dest_device.
        """
        src_block_ids = [block.block_id for block in blocks]
        self._allocators[src_device].swap_out(blocks)
        self._allocators[dst_device].swap_in(blocks)
        dst_block_ids = [block.block_id for block in blocks]

        current_swap_mapping: Dict[int, int] = {}
        for src_block_id, dst_block_id in zip(src_block_ids, dst_block_ids):
            if src_block_id is not None and dst_block_id is not None:
                self._swap_mapping[src_block_id] = dst_block_id
                current_swap_mapping[src_block_id] = dst_block_id
        return current_swap_mapping

    def get_num_full_blocks_touched(self, blocks: List[Block],
                                    device: BlockLocation) -> int:
        """Returns the number of full blocks that will be touched by
        swapping in/out the given blocks on to the 'device'.

        Args:
            blocks: List of blocks to be swapped.
            device (BlockLocation): BlockLocation to swap the 'blocks' on.

        Returns:
            int: the number of full blocks that will be touched by
                swapping in/out the given blocks on to the 'device'.
                Non full blocks are ignored when deciding the number
                of blocks to touch.
        """
        return self._allocators[device].get_num_full_blocks_touched(blocks)

    def clear_copy_on_writes(self) -> List[Tuple[int, int]]:
        """Clears the copy-on-write (CoW) state and returns the mapping of
            source to destination block IDs.

        Returns:
            List[Tuple[int, int]]: A list mapping source block IDs to 
                destination block IDs.
        """
        # CoW only supported on GPU
        device = BlockLocation.GPU
        return self._allocators[device].clear_copy_on_writes()

    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float) -> None:
        """Mark blocks as accessed, only use for prefix caching."""
        # Prefix caching only supported on GPU.
        device = BlockLocation.GPU
        return self._allocators[device].mark_blocks_as_accessed(block_ids, now)

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        """Mark blocks as accessed, only use for prefix caching."""
        # Prefix caching only supported on GPU.
        device = BlockLocation.GPU
        return self._allocators[device].mark_blocks_as_computed(block_ids)

    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        # Prefix caching only supported on GPU.
        device = BlockLocation.GPU
        return self._allocators[device].get_common_computed_block_ids(
            computed_seq_block_ids)

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        return frozenset(self._block_ids_to_allocator.keys())

    def get_prefix_cache_hit_rate(self, device: BlockLocation) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        assert device in self._allocators
        return self._allocators[device].get_prefix_cache_hit_rate()

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache for all devices."""
        success = True
        for allocator in self._allocators.values():
            success = success and allocator.reset_prefix_cache()
        return success

    def get_and_reset_swaps(self) -> List[Tuple[int, int]]:
        """Returns and clears the mapping of source to destination block IDs.
        Will be called after every swapping operations for now, and after every
        schedule when BlockManagerV2 become default. Currently not useful.

        Returns:
            List[Tuple[int, int]]: A mapping of source to destination block IDs.
        """
        mapping = self._swap_mapping.copy()
        self._swap_mapping.clear()
        return list(mapping.items())

    def find_cached_blocks_prefix(
        self,
        block_hashes: List[int],
        device: BlockLocation = BlockLocation.GPU,
    ) -> List[int]:
        return self._allocators[device].find_cached_blocks_prefix(block_hashes)

class ComputedBlocksTracker:
    """
    Tracks the computed blocks for each sequence.

    Internally, it maintains a map from sequence id to the list of block hashes
    for the sequence. We cache the hashes of the full blocks for each sequence,
    and make sure the hash is calculated in the same way as the allocator.
    When a sequence is being decoded, we also update the sequence's hash
    accordingly and incrementally.

    From the sequence hash, with prefix caching enabled, we could also calculate
    the number of cached tokens for the sequence by looking up the number of
    cached block hashes in the allocator.
    """

    # Note that we use 'None' as a string here instead of None because
    # as of Python 3.12, hash(None) returns a constant predictable value.
    # This could possibly make it easier to find and exploit hash
    # collisions. 'None' as a string will be hashed differently per process,
    # but consistently within the same process. This is the same as the
    # behavior of None prior to Python 3.12.
    _none_hash: int = hash('None')

    def __init__(
        self,
        allocator: DeviceAwareBlockAllocator,
        block_size: int,
        enable_caching: bool,
    ):
        self._allocator = allocator
        self._block_size = block_size
        self._enable_caching = enable_caching

        # A map from seq_id to the list of block hashes for the
        # sequence. This is so that we don't have to recompute the block hashes
        # for the sequence when we need to check if the sequence is cached.
        # Note a block that's not full will not have its hash calculated and
        # recorded.
        self._seq_id_to_blocks_hashes: Dict[int, List[int]] = {}

        # A map from seq_id to the number of tokens that are cached for the
        # sequence.
        # We need this so that a sequence in continuous prefill doesn't
        # accidentally see its cached token count change. See comments in
        # `get_num_cached_tokens` for more details.
        self._seq_id_to_num_tokens_computed: Dict[int, int] = {}


    def get_num_cached_tokens(self, seq: Request) -> int:
        if not self._enable_caching:
            return 0

        # We always try to update the sequence hashes on the fly.
        # This is to ensure that we don't miss any cached tokens for the
        # sequence during decode.
        # This routine should only update hash for any new blocks too.
        self._update_seq_hashes(seq)

        num_computed_tokens_prev = self._seq_id_to_num_tokens_computed.get(
            seq.seq_id, None)

        # TODO(rickyx): This hack could be removed once we mark blocks as
        # computed correctly with chunked prefills.
        if num_computed_tokens_prev is not None and seq.is_prefill():
            # For a sequence that is still in prefill, we don't
            # recompute the number of cached tokens.
            # This also handles correctly chunked prefill since currently
            # we mark blocks as computed even if the sequence is still partially
            # prefilled. So a continuously prefilled sequence should not
            # see its cached token count change while running.
            return num_computed_tokens_prev

        block_hashes = self._seq_id_to_blocks_hashes[seq.seq_id]

        # This is O(logN), where N is the number of blocks.
        num_cached_blocks = len(
            self._allocator.find_cached_blocks_prefix(block_hashes))
        num_cached_tokens = num_cached_blocks * self._block_size
        self._seq_id_to_num_tokens_computed[seq.seq_id] = num_cached_tokens
        return num_cached_tokens

    def remove_seq(self, seq_id: int) -> None:
        """Stop tracking the sequence."""
        if not self._enable_caching:
            return
        assert seq_id in self._seq_id_to_blocks_hashes
        del self._seq_id_to_blocks_hashes[seq_id]

        assert seq_id in self._seq_id_to_num_tokens_computed
        del self._seq_id_to_num_tokens_computed[seq_id]


class LastAccessBlocksTracker:
    """Manages the last access time of the tracked sequences, in order to allow
    an efficient update of allocator's block last access times
    """

    def __init__(self, allocator):
        self._allocator = allocator
        self._seq_last_access: Dict[int, Optional[float]] = {}

    def add_seq(self, seq_id: int) -> None:
        """Start tracking seq_id
        """
        assert seq_id not in self._seq_last_access
        self._seq_last_access[seq_id] = None

    def remove_seq(self, seq_id: int) -> None:
        """Stop tracking seq_id
        """
        assert seq_id in self._seq_last_access
        del self._seq_last_access[seq_id]

    def update_last_access(self, seq_id: int, time: float) -> None:
        assert seq_id in self._seq_last_access
        self._seq_last_access[seq_id] = time

    def update_seq_blocks_last_access(self, seq_id: int,
                                      block_ids: List[int]) -> None:
        assert seq_id in self._seq_last_access

        ts = self._seq_last_access[seq_id]

        if ts is None:
            # No last access was recorded, no need to update.
            return

        self._allocator.mark_blocks_as_accessed(block_ids, ts)

T = TypeVar("T")
def chunk_list(lst: List[T], chunk_size: int):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]