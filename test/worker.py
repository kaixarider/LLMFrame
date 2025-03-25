from MixFrame.worker.worker import Worker
from MixFrame.engine.single_stage_engine import PrefillEngine,DecodeEngine
from MixFrame.scheduler.prefill_stage_scheduler import PrefillSchedulerConfig
from MixFrame.scheduler.decode_stage_scheduler import DecodeSchedulerConfig
from MixFrame.config import ParallelConfig,ModelConfig,CacheConfig
from MixFrame.util import InferenceStage
import torch.distributed as dist

'''
Number of worker depends on parallel configuration.Each worker corresponds to a GPU.
A worker tests GPU memory ,load model in initiation.While in inference stage,it executes
calculation,migrate blocks and clear resources.This file tests functions of worker.
'''