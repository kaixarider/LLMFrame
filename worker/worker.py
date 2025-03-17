import torch.distributed as dist
import torch
from typing import Tuple
import logging
from MixFrame.config import DisParallelConfig,ParallelConfig,CacheConfig,ModelConfig,SchedulerConfig
from MixFrame.util import InferenceStage,set_random_seed,get_gpu_memory,MB,GB

logger = logging.getLogger(__name__)

class Worker:
    def __init__(self,
                 worker_id:int,
                 gpu_id:int,
                 stage:InferenceStage,
                 parallel_config:DisParallelConfig,
                 cache_config:CacheConfig,
                 sche_config:SchedulerConfig,
                 model_config:ModelConfig,
                 tp_rank:int,
                 pp_rank:int):
        self.worker_id=worker_id
        self.stage=stage
        self.parallel_config=parallel_config
        self.cache_config=cache_config
        self.schedule_config=sche_config
        self.tp_rank=tp_rank
        self.pp_rank=pp_rank
        self.gpu_id=gpu_id
        self.model_config=model_config
        self.k_cache=None
        self.v_cache=None
        
    def init_model(self)->None:
        set_random_seed(self.model_config.seed)
        
    def init_kv_cache_and_swap(self,num_gpu_blocks:int,num_cpu_blocks):
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
    
    def _get_block_size_in_bytes(
        self,
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        # the shape of one slot in k/v cache is [num_layers, num_local_heads, block_size, head_dim]
        num_layers = model_config.get_num_layers(parallel_config)
        num_heads = model_config.get_num_kv_heads(parallel_config)
        head_dim = model_config.get_head_size()

        key_cache_size = num_layers * num_heads * block_size * head_dim
        total = key_cache_size * 2
        dtype_size = model_config.get_dtype_size()
        return total * dtype_size
    
    @torch.inference_mode()
    def _profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # GPU and CPU blocks that can be allocated with the remaining free memory.

        # Profile memory usage with max_batch_size requests and the total
        # number of tokens equal to max_tokens_per_batch.
        total_gpu_memory = get_gpu_memory()
        peak_runtime_memory = (
            total_gpu_memory * 0.01
            + self.model_config.get_model_size_in_bytes(
                parallel_config=self.parallel_config
            )
        )
        logger.info(f"runtime peak memory: {peak_runtime_memory / GB:.3f} GB")
        logger.info(f"total GPU memory: {total_gpu_memory / GB:.3f} GB")
        block_size_in_bytes = self._get_block_size_in_bytes(
            block_size, self.model_config, self.parallel_config
        )
        logger.info(
            f"kv cache size for one token: {block_size_in_bytes / block_size / MB:.5f} MB"
        )
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_runtime_memory)
            // block_size_in_bytes
        )
        num_cpu_blocks = int(cpu_swap_space // block_size_in_bytes)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        logger.info(f"num_gpu_blocks: {num_gpu_blocks}")
        num_cpu_blocks = max(num_cpu_blocks, 0)
        logger.info(f"num_cpu_blocks: {num_cpu_blocks}")

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks