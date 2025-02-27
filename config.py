from typing import Optional
class ModelConfig:
    def __init__(self):
        return None

class CacheConfig: #config of kv cache
    def __init__(self,block_size: int,
        gpu_memory_utilization: float,
        swap_space: float,
        cache_dtype: str,
        is_attention_free: bool = False,
        num_gpu_blocks_override: Optional[int] = None,
        sliding_window: Optional[int] = None,
        enable_prefix_caching: bool = False,
        cpu_offload_gb: float = 0,
        calculate_kv_scales: Optional[bool] = None,)->None:
        self.gpu_memory_utilization = gpu_memory_utilization
        self.block_size = block_size
        self.swap_space = swap_space
        self.cache_dtype = cache_dtype
        self.is_attention_free = is_attention_free
        self.num_gpu_blocks_override = num_gpu_blocks_override
        self.sliding_window = sliding_window    
        self.enable_prefix_caching = enable_prefix_caching
        self.cpu_offload_gb = cpu_offload_gb
        self.calculate_kv_scales = calculate_kv_scales

        self.num_gpu_blocks:Optional[int] = None
        self.num_cpu_blocks:Optional[int] = None
class ParallelConfig:
    def __init__(self):
        return None
class QuantConfig:
    def __init__(self):
        return None