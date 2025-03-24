from abc import ABC,abstractmethod
from typing import List

import torch

from MixFrame.request.request import Request,BatchedRequests,MigrateRequest
from MixFrame.config import PrefillSchedulerConfig,ParallelConfig,CacheConfig
from MixFrame.block.blockmanager import BlockManager,AllocStatus
from MixFrame.util import BatchingType
    
class PrefillStageScheduler(ABC):
    '''Prefill stage scheduelr schedules requests to prefill,
    then it determines to decode locally,or migrate'''
    @abstractmethod
    def __init__(self,
                 parallel_config:ParallelConfig,
                 prefill_scheduler_config:PrefillSchedulerConfig,
                 cache_config:CacheConfig,
                 num_gpu_blocks:int,
                 num_cpu_blocks:int):
        '''initiate prefill stage scheduler'''
        assert prefill_scheduler_config.policy=='fcfs',"FCFS scheduler should be served for \
            fcfs policy!"
        self.parallel_config=parallel_config
        self.prefill_scheduler_config=prefill_scheduler_config
        self.block_manager=BlockManager(block_size=cache_config.block_size,num_gpu_blocks=num_gpu_blocks,
                                        num_cpu_blocks=num_cpu_blocks)# to be finished
        '''
        four queues.
        -waiting queue:requests waiting to be prefilled
        -running queue:requests that suit CB thus waiting for decoding
        -migrate_queue:finished prefilled but waiting to be decoded'''
        self.waiting_queue:List[Request]=[]
        self.running_queue:List[Request]=[]
        self.migrate_queue:List[MigrateRequest]=[]
        self.swap_queue=[]
    @abstractmethod
    def add_request(self,request:Request)->None:
        '''add_request to waiting queue'''
        raise NotImplementedError
    
    @abstractmethod
    def abort_request(self,request_id:int)->None:
        '''abort request that can't be executed'''
        raise NotImplementedError

    @abstractmethod
    def select_requests(self)->BatchedRequests:
        '''select requests for execution,prefill or continous batching'''
        raise NotImplementedError
    @staticmethod
    def _CB_or_PD(req:Request,sche_type:BatchingType)->None:
        '''determine whether continuous batching(CB) or Prefill Decode Disaggregation(PD)
        suit a batch'''
        req.schedule_type=sche_type
    @abstractmethod
    def _convert_request_to_Migrequest(self,request:Request):
        raise NotImplementedError
    @abstractmethod
    def clear_req(self,request:Request)->None:
        raise NotImplementedError
class FCFS_PrefillStageScheduler(PrefillStageScheduler):
    def __init__(self,
                 parallel_config:ParallelConfig,
                 prefill_scheduler_config:PrefillSchedulerConfig,
                 cache_config:CacheConfig,
                 num_gpu_blocks:int,
                 num_cpu_blocks:int)->None:
        super().__init__(parallel_config=parallel_config,prefill_scheduler_config=prefill_scheduler_config,
                         cache_config=cache_config,num_gpu_blocks=num_gpu_blocks,num_cpu_blocks=num_cpu_blocks)
    
    def add_request(self, request:Request)->None:
        self.waiting_queue.append(request)
        
        
    def abort_request(self, req:Request)->None:
        for (i,request) in enumerate(self.waiting_queue):
            if req.request_id==request.request_id:
                del self.waiting_queue[i]
                return
        for (i,request) in enumerate(self.running_queue):
            if req.request_id==request.request_id:
                self.block_manager.free(request)
                del self.running_queue[i]
                return
            
    def select_requests(self)->BatchedRequests:
        batch=BatchedRequests()
        #select from running queue first,then prefill requests in waiting queue
        for req in self.running_queue:
            if self.prefill_scheduler_config.max_batch_size>len(batch.requests) and \
                self.prefill_scheduler_config.max_token_num_each_req>req.get_len() and \
                self.block_manager.can_append_slots(req):
                    batch.add_request(req)
        for req in self.waiting_queue:
            if self.prefill_scheduler_config.max_batch_size>len(batch.requests) and \
                self.prefill_scheduler_config.max_token_num_each_req>req.get_len() and \
                self.block_manager.can_allocate(req)==AllocStatus.OK:
                    batch.add_request(req)
                    self.block_manager.allocate(req)
            if self.block_manager.can_allocate(req)==AllocStatus.NO or \
                not self.prefill_scheduler_config.max_batch_size>len(batch.requests) or\
                not self.prefill_scheduler_config.max_token_num_each_req>req.get_len():
                    self.abort_request(req)
        return batch
    
    def _convert_request_to_Migrequest(self,req:Request)->None:
        #convert request to migration request and add it to migrate_queue
        block_table=self.block_manager.req_table[req.request_id]
        migrate_request=MigrateRequest(req=req,para_config=self.parallel_config)
        blocks=block_table.used_blocks()
        for block in blocks:
            token_ids=block._token_ids
            migrate_request.add_block_token_ids(token_ids)
        self.block_manager.free(req)
        self.migrate_queue.append(migrate_request)
        return 
    
    def clear_req(self, req:Request)->None:
        self.block_manager.free(req)

def get_prefill_scheduler(sche_config:PrefillSchedulerConfig,
                       parallel_config:ParallelConfig,
                       cache_config:CacheConfig,
                       num_gpu_blocks:int,
                       num_cpu_blocks:int)->PrefillStageScheduler:
    match sche_config.policy:
        case "fcfs":
            return FCFS_PrefillStageScheduler(parallel_config=parallel_config,
                                              prefill_scheduler_config=sche_config,
                                              cache_config=cache_config,
                                              num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)
        case _:
            raise ValueError("no such prefill schedule policy")