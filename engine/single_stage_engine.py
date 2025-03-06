'''the basic type of engine.Engine is based on scheduler and schedule requests according to
schedule policy'''
from abc import ABC,abstractmethod

from MixFrame.config.parallel_config import ParallelConfig
from MixFrame.config.scheduler_config import DecodeScheduleConfig,PrefillScheduleConfig
from MixFrame.request.request import Request

class SingleStepEngine(ABC):
    @abstractmethod
    def add_reqeust(req:Request):
        NotImplementedError