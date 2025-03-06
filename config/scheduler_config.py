from abc import ABC,abstractmethod
from enum import Enum


from MixFrame.request.request import BatchedRequests,Request,ScheduleType,MigrateRequests
from MixFrame.config.parallel_config import ParallelConfig
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
        assert policy in ['fcfs'],"prefill scheduler \
        only supports fcfs policy"
        self.policy=policy
        self.max_batch_size=max_batch_size
        self.max_token_num_each_req=max_token_num_each_req
        self.prefill_IDF=None #supposed to be a json to predict prefill duration
    def set_IDF(self,IDF):
        self.IDF=IDF
    
    