from enum import Enum
from typing import Callable,List,Optional,Dict


from MixFrame.config import CacheConfig,ModelConfig,ParallelConfig
from MixFrame.request.request import Request
from block_allocator import BlockLocation,BlockList,CpuGpuBlockAllocator,ComputedBlocksTracker,LastAccessBlocksTracker
from block_table import BlockTable,BlockAllocator
SeqId=int
EncoderSeqId=int
class ManagerType(Enum):
    '''type of blockmanage
    -type one generates kv cache ant utilize it(if CB is taken)
    -type two takes migration and executes decode'''
    TypeOne=1,
    TypeTwo=2
class BlockManager:
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,       
    )->None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )
        
        self.block_tables: Dict[SeqId, BlockTable] = {}
        self.cross_block_tables: Dict[EncoderSeqId, BlockTable] = {}

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.block_allocator, self.block_size, self.enable_caching)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.block_allocator)
    
   