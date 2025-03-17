'''the basic type of engine.Engine is based on scheduler and schedule requests according to
schedule policy'''
from abc import ABC,abstractmethod
from typing import Tuple,List
import asyncio

from MixFrame.config import ParallelConfig,DisParallelConfig,ModelConfig
from MixFrame.config import DecodeSchedulerConfig,PrefillSchedulerConfig,SchedulerConfig
from MixFrame.scheduler.decode_stage_scheduler import DecodeStageScheduler,FCFS_DecodeStageScheduler
from MixFrame.scheduler.prefill_stage_scheduler import PrefillStageScheduler, FCFS_PrefillStageScheduler
from MixFrame.request.request import Request,MigrateRequests
from MixFrame.worker.worker import Worker
from MixFrame.tokenizer.tokenizer import get_tokenizer

class SingleStepEngine(ABC):
    @abstractmethod
    def getscheduler(self,sche_config:SchedulerConfig)->(DecodeStageScheduler|PrefillStageScheduler):
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
    def __init__(self,prefill_sche_config:PrefillSchedulerConfig,para_config:DisParallelConfig,model_config)->None:
        self.sche_config=prefill_sche_config
        self.para_config=para_config
        self.prefill_scheduler=self.getscheduler(prefill_sche_config)
    
    async 

    
class DecodeEngine(SingleStepEngine):
    def __init__(self,decode_sche_config:DecodeSchedulerConfig,para_config:DisParallelConfig,model_config)->None:
        self.sche_config=decode_sche_config
        self.para_config=para_config
        self.decode_scheduler=self.getscheduler(decode_sche_config)
        
    def add_request(self,Mig_request:MigrateRequests):
        self.decode_scheduler.add_request(Mig_request)
            
class CoTEngine(SingleStepEngine):
    def __init__(self,CoT_sche_config,para_config,model_config)->None:
        return
    
class ResponseEngine(SingleStepEngine):
    def __init__(self,Response_sche_config,para_config,model_config)->None:
        return
