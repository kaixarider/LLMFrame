from typing import Optional,Union,Tuple,Dict
from pathlib import Path
import os
from os import PathLike
from transformers import PretrainedConfig,AutoConfig
from enum import Enum
from abc import ABC,abstractmethod
import huggingface_hub
from huggingface_hub import (file_exists, hf_hub_download,
                             try_to_load_from_cache)
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME
from MixFrame.request.request import BatchedRequests,Request,ScheduleType,MigrateRequests
import logging
logger = logging.getLogger(__name__)
'''include CacheConfig,ModelConfig,ParallelConfig and SchedulerConfig'''
class CacheConfig:
    """Configuration for the key-value cache.

    Args:
        block_size: Number of tokens in a block.
        max_num_blocks_per_req: Maximum number of blocks each request can have.
        gpu_memory_utilization: The maximum percentage of GPU memory that can be used.
        cpu_swap_space: The maximum CPU swap space in bytes that can be used.
    """
    def __init__(
        self,
        block_size:int,
        max_num_blocks_per_req:int,
        gpu_memory_utilization:float=0.9,
        cpu_swap_space:int=0
    )->None:
        self.block_size=block_size
        self.max_num_blocks_per_req=max_num_blocks_per_req
        self.gpu_memory_utilization=gpu_memory_utilization
        self.cpu_swap_space=cpu_swap_space
    
    def get_num_blocks(self):
        self.num_gpu_blocks=
        self.num_cpu_blocks=
class ParallelConfig:
    '''config numbers of parallel nodes'''
    def __init__(
      self,
      tp_size:int=1,#tensor_parallel
      tp_rank:int=0,
      dp_size:int=1,#data_parallel
      dp_rank:int=0
    )->None:
        self.tp_size=tp_size
        self.tp_rank=tp_rank
        self.dp_size=dp_size
        self.dp_rank=dp_rank
        
        self.world_size=dp_size*tp_size
        self.use_parallel=self.world_size>1
    
    def show_config(self)->Dict:
      return {"tp_size":self.tp_size,
              "tp_rank":self.tp_rank,
              "dp_size":self.dp_size,
              "dp_rank":self.dp_size}
    
class DisParallelConfig:
    def __init__(
      self,
      PrefillParallelConfig:ParallelConfig,
      DecodeParallelConfig:ParallelConfig
    )->None:
      self.prefill_parallel_config=PrefillParallelConfig
      self.decode_parallel_config=DecodeParallelConfig
      
    def num_workers(self):
      return self.decode_parallel_config.world_size+self.prefill_parallel_config.world_size  
class ModelConfig:
    """Configuration for the model.

Args:
    model: Model name or path.
    tokenizer: Tokenizer name or path.
    tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
        available, and "slow" will always use the slow tokenizer.
        Default to "auto".
    trust_remote_code: Trust remote code (e.g., from HuggingFace) when
        downloading the model and tokenizer.
    dtype: Data type of the model. Default to "fp16".
    seed: Random seed for reproducing.
"""
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str],
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        dtype: str = "fp16",
        seed: int = 1,
        use_dummy_weights: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer if tokenizer else model
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.seed = seed
        self.hf_config=self._get_hf_config()
        self.hf_text_config=self._get_hf_text_config(self.hf_config)
        sliding_window = getattr(self.hf_text_config, "sliding_window", None)
    def _get_hf_config(self):
        try:
            config = AutoConfig.from_pretrained(
                self.model, trust_remote_code=self.trust_remote_code
            )
        except:
            raise ValueError(
                f"Failed to load the model config, please check the model name or path: {self.model}"
            )
        return config
    
    def _get_hf_text_config(config: PretrainedConfig):
        """Get the "sub" config relevant to llm for multi modal models.
        No op for pure text models.
        """
        if hasattr(config, "text_config"):
            # The code operates under the assumption that text_config should have
            # `num_attention_heads` (among others). Assert here to fail early
            # if transformers config doesn't align with this assumption.
            assert hasattr(config.text_config, "num_attention_heads")
            return config.text_config
        else:
            return config
    
    def get_vocab_size(self) -> int:
        return self.hf_text_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.hf_text_config.hidden_size
    
    @property
    def is_deepseek_mla(self) -> bool:
        return (hasattr(self.hf_text_config, "model_type")) \
                and (self.hf_text_config.model_type in \
                    ('deepseek_v2', 'deepseek_v3'))\
                and (self.hf_text_config.kv_lora_rank is not None)


    def get_head_size(self) -> int:
        # TODO remove hard code
        if self.is_deepseek_mla:
            qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim",
                                       0)
            if self.use_mla:
                return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config,
                                           "qk_nope_head_dim", 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                    return qk_rope_head_dim + qk_nope_head_dim

        if self.is_attention_free:
            return 0

        if hasattr(self.hf_text_config, "head_dim"):
            return self.hf_text_config.head_dim
        # FIXME(woosuk): This may not be true for all models.
        return (self.hf_text_config.hidden_size //
                self.hf_text_config.num_attention_heads)

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False))
        if not new_decoder_arch_falcon and getattr(self.hf_text_config,
                                                   "multi_query", False):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        # For DBRX and MPT
        if self.hf_config.model_type == "mpt":
            if "kv_n_heads" in self.hf_config.attn_config:
                return self.hf_config.attn_config["kv_n_heads"]
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type == "dbrx":
            return getattr(self.hf_config.attn_config, "kv_n_heads",
                           self.hf_config.num_attention_heads)

        if self.is_attention_free:
            return 0

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_text_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
        """Returns the number of KV heads per GPU."""
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1

        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1,
                   total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(self,
                                parallel_config: "ParallelConfig") -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // parallel_config.tensor_parallel_size

    def get_layers_start_end_indices(
            self, parallel_config: "ParallelConfig") -> Tuple[int, int]:
        from vllm.distributed.utils import get_pp_indices
        total_num_hidden_layers = getattr(self.hf_text_config,
                                          "num_hidden_layers", 0)
        pp_rank = parallel_config.rank // parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        start, end = self.get_layers_start_end_indices(parallel_config)
        return end - start

    @property
    def use_mla(self) -> bool:
        if not self.is_deepseek_mla:
            return False

        if self.quantization is not None and self.quantization not in [\
            "fp8", "compressed-tensors"]:
            logger.warning(
                "MLA is not supported with %s quantization. "
                "Disabling MLA.", self.quantization)
            return False

        # If using a "compressed-tensors" checkpoint, check that all groups
        # have fp8 for both weights and activations.
        if self.quantization == "compressed-tensors":
            quant_config = self._parse_quant_hf_config()
            for group_name, cfg in quant_config.get("config_groups", {
                    "": {}
            }).items():
                act_cfg = cfg.get("input_activations", {})
                act_type = None if act_cfg is None else act_cfg.get("type", "")
                w_cfg = cfg.get("weights", {})
                w_type = None if w_cfg is None else w_cfg.get("type", "")
                if act_type != "fp8" or w_type != "fp8":
                    logger.warning(
                        "compressed-tensors MLA support requires fp8 "
                        "activations and weights in group '%s', but got "
                        "activations type '%s' and weights type '%s'.\n "
                        "Full config: %s", group_name, act_type, w_type,
                        quant_config)
                    return False

        return True


    
  
class SchedulerConfig(ABC):
    '''config scheduler,including schedule policy,parallel config,up bound of scheduler'''
    @abstractmethod
    def __init__(self,
                 parallel_config:ParallelConfig,
                 policy:str,
                 max_batch_size:int,
                 max_token_num_each_req:int):
        raise NotImplementedError()
    
class PrefillSchedulerConfig(SchedulerConfig):
    def __init__(self, 
                 parallel_config, 
                 policy, 
                 max_batch_size, 
                 max_token_num_each_req):
        self.parallel_config=parallel_config
        assert policy in ['fcfs'],"prefill scheduler \
        only supports fcfs policy"
        self.policy=policy
        self.max_batch_size=max_batch_size
        self.max_token_num_each_req=max_token_num_each_req
        self.prefill_IDF=None #supposed to be a json to predict prefill duration
    def set_IDF(self,IDF):
        self.IDF=IDF

class DecodeSchedulerConfig(SchedulerConfig):
    def __init__(self, 
                 parallel_config, 
                 policy, 
                 max_batch_size, 
                 max_token_num_each_req):
        self.parallel_config=parallel_config
        assert policy in ['fcfs'],f"Error!Scheduler doesn't support {policy} "
        self.policy=policy
        self.max_batch_size=max_batch_size
        self.max_token_num_each_req=max_token_num_each_req
        self.prefill_IDF=None #supposed to be a json to predict prefill duration
    def set_IDF(self,IDF):
        self.IDF=IDF
    
