
from typing import Optional,Tuple,Dict,List
import enum
from MixFrame.block.blocktable import BlockTable,BlockAllocator,BlockLocation
from MixFrame.request.request import Request,BatchedRequests,MigrateRequest
ReqId=int
BlockId=int
class AllocStatus(enum.Enum):
    OK=enum.auto()
    LATER=enum.auto()
    NO=enum.auto()
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
        self.gpu_block_allocator=self.create_allocator(num_blocks=num_gpu_blocks,location=BlockLocation.GPU,block_size=block_size)
        self.cpu_block_allocator=self.create_allocator(num_blocks=num_cpu_blocks,location=BlockLocation.CPU,block_size=block_size)
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
        #print(self.watermark_blocks)
        self.req_table:Dict[ReqId,BlockTable]={}#Record the blocks a req use.Migrate corresponding blocks
    def can_allocate(self,req:Request,ahead_slots:int=0)->AllocStatus:
        num_required_blocks = BlockTable.get_num_required_blocks(req._update_cached_all_tokens(),self.block_size,ahead_slots)
        num_free_gpu_blocks = self.gpu_block_allocator.get_num_free_block()
        #print(f"test req is {req.request_id},required block num is {num_required_blocks},num free block is {num_free_gpu_blocks}")
        if (self.num_total_gpu_blocks - num_required_blocks
                < self.watermark_blocks):
            return AllocStatus.NO
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER
    def can_allocate_Migreq(self,Migreq:MigrateRequest,ahead_slots:int=0)->AllocStatus:
        num_required_blocks = BlockTable.get_num_required_blocks(Migreq.req._update_cached_all_tokens(),self.block_size,ahead_slots)
        num_free_gpu_blocks = self.gpu_block_allocator.get_num_free_block()
        if (self.num_total_gpu_blocks - num_required_blocks
                < self.watermark_blocks):
            return AllocStatus.NO
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER
    def _get_req_blocktable(self,req:Request)->BlockTable:
        block_table = BlockTable(
            block_size=self.block_size,block_allocator=self.gpu_block_allocator,max_block_sliding_window=self.sliding_window
        )
        if req.get_input_token_ids():
            block_table.allocate(req.get_input_token_ids())
        return block_table
    def _get_Migreq_blocktable(self,req:MigrateRequest)->BlockAllocator:
        block_table = BlockTable(
            block_size=self.block_size,block_allocator=self.gpu_block_allocator,max_block_sliding_window=self.sliding_window
        )
        blocks=block_table._copy_blocks(req.blocks,location=BlockLocation.GPU)
        block_table.update(blocks)
        print(f'block table of migrate request is{block_table.used_blocks()}')
        return block_table
    
    def allocate(self,req:Request)->None:
        assert not req.request_id in self.req_table.keys(),f"block table of {req.request_id} exists"
        block_table=self._get_req_blocktable(req)
        self.req_table[req.request_id]=block_table
        
    def allocate_prefilled_req(self,req:MigrateRequest):
        assert not req.req.request_id in self.req_table.keys(),f"block table of {req.req.request_id} exists"
        block_table=self._get_Migreq_blocktable(req)
        self.req_table[req.req.request_id]=block_table
        
    def can_append_slots(self,req:Request,ahead_slots)->bool:
        num_touched_blocks=0
        block_table=self.req_table[req.request_id]
        num_touched_blocks += (
                block_table.get_num_blocks_touched_by_append_slots(
                    token_ids=block_table.get_unseen_token_ids(
                        req._cached_all_token_ids),
                    num_lookahead_slots=ahead_slots,
                ))

        num_free_gpu_blocks = self.gpu_block_allocator.get_num_free_block()
        return num_touched_blocks <= num_free_gpu_blocks
        
    def append_slots(
        self,
        req: Request,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        block_table = self.req_table[req.request_id]
        if self.can_append_slots(req):
            block_table.append_token_ids(
                token_ids=block_table.get_unseen_token_ids(req._cached_all_token_ids),
                num_lookahead_slots=num_lookahead_slots,
                num_computed_slots=req._num_computed_tokens,
            )
    def create_allocator(
        self,
        num_blocks:int,
        location:BlockLocation,
        block_size:int
    )->BlockAllocator:
        block_allocator=BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
            location=location,
        )
        return block_allocator

    def free(self, req:Request) -> None:
        req_id = req.request_id

        if req_id not in self.req_table:
            # Already freed or haven't been scheduled yet.
            return

        # Update seq block ids with the latest access time

        # Free table/blocks
        self.req_table[req_id].free()
        del self.req_table[req_id]
        
    def swap_out(self,req:Request)->Optional[List[Tuple[int,int]]]:
        physical_block_id_mapping = []
        blocks=self.req_table[req.request_id].blocks
        if len(blocks)==0:
            return None
        cpu_block_ids=self.cpu_block_allocator.swap_in(blocks)
        gpu_block_ids=self.gpu_block_allocator.swap_out(blocks)
        self.req_table[req.request_id].update(blocks)
        seq_physical_block_id_mapping = {
                self.gpu_block_allocator.get_physical_block_id(gpu_block_ids[i]):
                self.cpu_block_allocator.get_physical_block_id(cpu_block_ids[i])
                for i in range(len(gpu_block_ids))
            }
        physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        return physical_block_id_mapping
    
    def swap_in(self, req:Request) -> Optional[List[Tuple[int, int]]]:
        """Returns the block id mapping (from CPU to GPU) generated by
        swapping in the given seq_group with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from CPU 
                to GPU.
        """
        physical_block_id_mapping = []
        blocks = self.req_table[req.request_id].blocks
        if len(blocks) == 0:
            return
        cpu_block_ids=self.cpu_block_allocator.swap_out(blocks)
        gpu_block_ids=self.gpu_block_allocator.swap_in(blocks)

        # Refresh the block ids of the table (post-swap)
        self.req_table[req.request_id].update(blocks)

        seq_physical_block_id_mapping = {
            self.cpu_block_allocator.get_physical_block_id(cpu_block_ids[i]):
                self.gpu_block_allocator.get_physical_block_id(gpu_block_ids[i])
                for i in range(len(gpu_block_ids))
        }

        physical_block_id_mapping.extend(
            list(seq_physical_block_id_mapping.items()))

        return physical_block_id_mapping