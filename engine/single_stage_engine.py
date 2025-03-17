'''the basic type of engine.Engine is based on scheduler and schedule requests according to
schedule policy'''
from abc import ABC,abstractmethod
from typing import Tuple,List
import asyncio
import enum
from MixFrame.config import ParallelConfig,DisParallelConfig,ModelConfig,CacheConfig
from MixFrame.config import DecodeSchedulerConfig,PrefillSchedulerConfig,SchedulerConfig
from MixFrame.scheduler.decode_stage_scheduler import DecodeStageScheduler,get_FCFS_decode_scheduler
from MixFrame.scheduler.prefill_stage_scheduler import PrefillStageScheduler, get_FCFS_prefill_scheduler
from MixFrame.request.request import Request,MigrateRequest
from MixFrame.worker.worker import Worker
from MixFrame.tokenizer.tokenizer import get_tokenizer
class SchedulerType(enum.Enum):
    FCFS=enum.auto()
class SingleStepEngine(ABC):
    @abstractmethod
    def getscheduler(self)->(DecodeStageScheduler|PrefillStageScheduler):
        raise NotImplementedError
    @abstractmethod
    def load_model(self,model_config:ModelConfig):
        raise NotImplementedError
    @abstractmethod
    async def recv_request(self):
        raise NotImplementedError
    @abstractmethod
    def __init__(self,sche_config:SchedulerConfig,para_config:DisParallelConfig)->None:
        raise NotImplementedError
    

    
    
class PrefillEngine(SingleStepEngine):
    def __init__(self,prefill_sche_config:PrefillSchedulerConfig,para_config:DisParallelConfig,
                 model_config:ModelConfig,cache_config:CacheConfig)->None:
        self.sche_config=prefill_sche_config
        self.para_config=para_config
        self.cache_config=cache_config
        self.prefill_scheduler=self.getscheduler()
        self.model_config=model_config
    def getscheduler(self,type:SchedulerType):
        match type:
            case SchedulerType.FCFS:
                return get_FCFS_prefill_scheduler(sche_config=self.sche_config,parallel_config=self.para_config,cache_config=self.cache_config)
            case _:
                raise TypeError(f"There is no such {type} schedule type!")
    def load_model(self,model_config:ModelConfig):
        return 
    async def recv_request(self):
        return
    
class DecodeEngine(SingleStepEngine):
    def __init__(self,prefill_sche_config:DecodeSchedulerConfig,para_config:DisParallelConfig,
                 model_config:ModelConfig,cache_config:CacheConfig)->None:
        self.sche_config=prefill_sche_config
        self.para_config=para_config
        self.cache_config=cache_config
        self.decode_scheduler=self.getscheduler()
        self.model_config=model_config
    def getscheduler(self,type:SchedulerType):
        match type:
            case SchedulerType.FCFS:
                return get_FCFS_decode_scheduler(sche_config=self.sche_config,parallel_config=self.para_config,cache_config=self.cache_config)
            case _:
                raise TypeError(f"There is no such {type} schedule type!")
            
class CoTEngine(SingleStepEngine):
    def __init__(self,CoT_sche_config,para_config,model_config)->None:
        return
    
class ResponseEngine(SingleStepEngine):
    def __init__(self,Response_sche_config,para_config,model_config)->None:
        return
