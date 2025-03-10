from MixFrame.config import CacheConfig,ModelConfig,ParallelConfig
from MixFrame.request.request import Request
from enum import Enum
from typing import Callable,List
class BlockLocation(Enum):
    CPU="CPU"
    GPU="GPU"
class ManagerType(Enum):
    '''type of blockmanage
    -type one generates kv cache ant utilize it(if CB is taken)
    -type two takes migration and executes decode'''
    TypeOne=1,
    TypeTwo=2
class BlockManager:
    def __init__(self,
                 manager_type:ManagerType,
                 cache_config:CacheConfig,
                 model_config:ModelConfig,
                 parallel_config:ParallelConfig,
                 max_cpu_blocks_num:int,
                 max_gpu_blocks_num:int,
                 engine_remote_call_all_workers_async:Callable#??
                 ):
        self.manager_type=manager_type
        self.cache_config=cache_config
        self.model_config=model_config
        self.parallel_config=parallel_config
        self.max_cpu_blocks_num=max_cpu_blocks_num
        self.max_gpu_blocks_num=max_gpu_blocks_num
        self.engine_remote_call_all_workers_async=engine_remote_call_all_workers_async
        
        self.block_table={}
        self.free_gpu_block_list=list(range(max_gpu_blocks_num))
        self.free_cpu_block_list=list(range(max_cpu_blocks_num))
        self.request_location={}
        self.swapping_cpu_list=[]
        self.swapping_gpu_list=[]
    
    def num_avail_gpu_blocks(self)->int:
        return len(self.free_gpu_block_list)+len(self.swapping_gpu_list)
    
    def num_avail_cpu_blocks(self)->int:
        return len(self.free_cpu_block_list)+len(self.swapping_cpu_list)
    
    def _get_free_blocks(self,location:BlockLocation,needed_blocks_num:int)->List[int]:
        match location:
            case BlockLocation.GPU:
                num_avail_blocks=self.num_avail_gpu_blocks()
                assert(needed_blocks_num<=num_avail_blocks),\
                "Error!There is no enough GPU blocks!"
                if (num_avail_blocks>len(self.free_gpu_block_list)):
                    self.engine_remote_call_all_workers_async("wait_for_all_swap_out")
                    self.free_gpu_blocks_list += self.swapping_gpu_blocks_list
                    self.swapping_gpu_blocks_list = []
                blocks = self.free_gpu_blocks_list[:needed_blocks_num]
                self.free_gpu_blocks_list = self.free_gpu_blocks_list[needed_blocks_num:]
                return blocks
            case BlockLocation.CPU:
                num_avail_blocks=self.num_avail_cpu_blocks()
                assert(needed_blocks_num<=num_avail_blocks),\
                "Error!There is no enough CPU blocks!"
                if (num_avail_blocks>len(self.free_cpu_block_list)):
                    self.engine_remote_call_all_workers_async("wait_for_all_swap_om")
                    self.free_cpu_blocks_list += self.swapping_cpu_blocks_list
                    self.swapping_cpu_blocks_list = []
                blocks = self.free_cpu_blocks_list[:needed_blocks_num]
                self.free_cpu_blocks_list = self.free_cpu_blocks_list[needed_blocks_num:]
            case _:
                print("Error!Wrong location!")
    
    def allocate(self,num_blocks_needed:int)