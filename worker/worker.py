import torch.distributed as dist
import torch
from typing import Tuple,List
import logging
import time
import enum

from typing import Dict,List
from MixFrame.config import DisParallelConfig,ParallelConfig,CacheConfig,ModelConfig,SchedulerConfig
from MixFrame.util import InferenceStage,set_random_seed,get_gpu_memory,MB,GB
from MixFrame.request.request import Request
from MixFrame.util import InferenceStage
Req_id=int
duration=float
logger = logging.getLogger(__name__)
class Worker:
    def __init__(self,
                 gpu_id:int,
                 stage:InferenceStage,
                 parallel_config:ParallelConfig,
                 cache_config:CacheConfig,
                 sche_config:SchedulerConfig,
                 model_config:ModelConfig):
        self.stage=stage
        self.parallel_config=parallel_config
        self.cache_config=cache_config
        self.schedule_config=sche_config
        self.gpu_id=gpu_id
       
        tp_size=parallel_config.tp_size
        pp_size=parallel_config.pp_rank
        tp_rank=parallel_config.tp_rank
        pp_rank=parallel_config.pp_rank

        self.worker_id=tp_size*pp_rank+tp_rank
        self.tp_size=tp_size
        self.pp_size=pp_size
        self.world_size=tp_size*pp_size
        self.model_config=model_config
        self.k_cache=None
        self.v_cache=None
        self.swap_event_table:Dict[int,torch.cuda.Event]={}
        #profile
        self.prefill_duration:dict[Req_id,duration]={}
        self.decode_duration:dict[Req_id,duration]={}
        self.sample:dict[Req_id,duration]={}
        
    def init_model(self)->None:
        set_random_seed(self.model_config.seed)
        #include load model weight and init parallel environment
        
    def init_kv_cache_and_swap(self):
        num_gpu_blocks,num_cpu_blocks=self._profile_num_available_blocks()
        kv_cache_shape = (
            num_gpu_blocks,
            self.model_config.get_num_layers(self.parallel_config),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.cache_config.block_size,
            self.model_config.get_head_size(),
        )
        self.k_cache = torch.empty(
            kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        self.v_cache = torch.empty(
            kv_cache_shape, dtype=self.model_config.get_torch_dtype(), device="cuda"
        )
        # kv swap is [num_cpu_blocks, num_layers, num_local_heads, block_size, head_dim]
        # We pin memory here in order to leverage cudaMemcpyAsync when swapping
        kv_swap_shape = (num_cpu_blocks,) + kv_cache_shape[1:]
        self.k_swap = torch.empty(
            kv_swap_shape, dtype=self.model_config.get_torch_dtype(), device="cpu", pin_memory=True
        )
        self.v_swap = torch.empty(
            kv_swap_shape, dtype=self.model_config.get_torch_dtype(), device="cpu", pin_memory=True
        )
        torch.cuda.synchronize()
    
    def _profile_num_available_blocks(
        self,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # GPU and CPU blocks that can be allocated with the remaining free memory.

        # Profile memory usage with max_batch_size requests and the total
        # number of tokens equal to max_tokens_per_batch.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()
        memory_for_current_instance = total_gpu_memory * \
            self.cache_config.gpu_memory_utilization
        cache_block_size=self.cache_config.get_block_size_in_bytes()
        if cache_block_size == 0:
            num_gpu_blocks = 0
            num_cpu_blocks = 0
        else:
            num_gpu_blocks = int(memory_for_current_instance // cache_block_size)
            num_cpu_blocks = int(cpu_swap_space//
                                 cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        return num_gpu_blocks, num_cpu_blocks

    def forward(
        self,
        request_ids: List[int],
        input_tokens_batched,
        first_token_indexes,
        block_table,
    ) -> List[int]:
        """Run one step of inference on the batch of requests."""

        start = time.time()
        # Check whether synchronization is necessary
        for request_id in request_ids:
            if request_id in self.swap_event_table:
                # We let the current stream wait for the swap event
                # This is non-blocking (It just stop the current stream instead
                # of chocking the CPU)
                self.swap_event_table[request_id].wait(torch.cuda.current_stream())
                self.swap_event_table.pop(request_id, None)
        self.blocked_swapping_time += time.time() - start

        start = time.time()
        # print(f"Worker {self.stage}.#{self.worker_id} Step begin")
        # run forward
        generated_tokens_ids = self.model.forward(
            input_tokens_batched,
            first_token_indexes,
            self.k_cache,
            self.v_cache,
            block_table,
        )
        execution_time = time.time() - start
        self._profile_execution_time(execution_time=execution_time,request_ids=request_ids)
        return generated_tokens_ids
    
    def _profile_execution_time(self,execution_time:float,request_ids:List[Req_id]):
        for req_id in request_ids:
            if self.stage==InferenceStage.prefill:
                if req_id in self.prefill_duration:
                    self.prefill_duration[req_id]+=execution_time
                else:
                    self.prefill_duration[req_id]=execution_time
            else:
                if req_id in self.decode_duration:
                    self.decode_duration[req_id]+=execution_time
                else:
                    self.decode_duration[req_id]=execution_time

    
    def init_distributed_environment(
        self,
        distributed_init_method: str = "env://",
        local_rank: int = -1,
        backend: str = "nccl",
    ):
        logger.debug(
            "world_size=%d rank=%d local_rank=%d "
            "distributed_init_method=%s backend=%s", self.world_size, self.worker_id, local_rank,
            distributed_init_method, backend)
        if not torch.distributed.is_initialized():
            assert distributed_init_method is not None, (
                "distributed_init_method must be provided when initializing "
                "distributed environment")
            # this backend is used for WORLD
            torch.distributed.init_process_group(
                backend=backend,
                init_method=distributed_init_method,
                world_size=self.world_size,
                rank=self.worker_id)


        
        