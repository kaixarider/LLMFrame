
from typing import Optional,Tuple,Dict

from MixFrame.block.blocktable import BlockTable,BlockAllocator,BlockLocation
from MixFrame.request.request import Request,BatchedRequests
ReqId=int
BlockId=int
class BlockManager:
    def __init__(
        self,
        block_size:int,
        num_gpu_blocks:int,
        num_cpu_blocks:int,
        watermark:float=0.01,
        sliding_window:Optional[int]=None
    ):
        self.block_size=block_size
        self.num_total_gpu_blocks=num_gpu_blocks
        self.num_total_cpu_blocks=num_cpu_blocks
        
        self.sliding_window = sliding_window
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
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        cpu_block_allocator,gpu_block_allocator=self.create_allocator((num_gpu_blocks,num_cpu_blocks,block_size))
        self.req_table:Dict[ReqId,BlockTable]={}#Record the blocks a req use.Migrate corresponding blocks
        
        
    
    def create_allocator(
        self,
        num_gpu_blocks:int,
        num_cpu_blocks:int,
        block_size:int
    )->Tuple[BlockAllocator,BlockAllocator]:
        block_ids=list(range(num_gpu_blocks+num_cpu_blocks))
        gpu_block_ids = block_ids[:num_gpu_blocks]
        cpu_block_ids = block_ids[num_gpu_blocks:]
        cpu_block_allocator=BlockAllocator(num_cpu_blocks,block_size,BlockLocation.CPU,cpu_block_ids)
        gpu_block_allocator=BlockAllocator(num_gpu_blocks,block_size,BlockLocation.GPU,gpu_block_ids)
        return tuple(cpu_block_allocator,gpu_block_allocator)