from abc import ABC,abstractmethod
from typing import List
import torch
from MixFrame.request.request import Request,BatchedRequests,BatchingType,MigrateRequest,RequestStatus
from MixFrame.config import DecodeSchedulerConfig,ParallelConfig,CacheConfig
from MixFrame.block.blockmanager import BlockManager
class DecodeStageScheduler(ABC):
    '''Prefill stage scheduelr schedules requests to prefill,
    then it determines to decode locally,or migrate'''
    @abstractmethod
    def __init__(self,
                 parallel_config:ParallelConfig,
                 decode_scheduler_config:DecodeSchedulerConfig):
        '''initiate prefill stage scheduler'''
        raise NotImplementedError
    @abstractmethod
    async def recv_mig_request(self):
        raise NotImplementedError
    @abstractmethod
    def _add_request(self,Mig_req:MigrateRequest)->None:
        '''add_request to waiting queue'''
        raise NotImplementedError
    
    @abstractmethod
    def abort_request(self,request:Request)->None:
        '''abort request that can't be executed'''
        raise NotImplementedError
    
    @abstractmethod
    def schedule_requests(self)->BatchedRequests:
        '''select requests for execution,prefill or continous batching'''
        raise NotImplementedError

    @abstractmethod
    def _convert_migrate_requests(self,Mig_request:MigrateRequest)->Request:
        raise NotImplementedError
    
class FCFS_DecodeStageScheduler(DecodeStageScheduler):
    def __init__(self,parallel_config:ParallelConfig,
                 decode_scheduler_config:DecodeSchedulerConfig,
                 cache_config:CacheConfig):
        self.parallel_config=parallel_config
        self.decode_scheduler_config=decode_scheduler_config 
        self.block_manager=BlockManager(block_size=cache_config.block_size,num_gpu_blocks=cache_config.num_gpu_blocks,num_cpu_blocks=cache_config.num_cpu_blocks)
        #queues
        
        self.waiting_queue:List[MigrateRequest]=[] #queues containing migrated requests that haven't been transformed to request
        self.running_queue:List[Request]=[] #queues containing decoding requests
        self.swap_queue:List[Request]=[]
    async def recv_mig_request(self):
        mig_req=torch.ops.recv()
        self.waiting_queue.append(mig_req)
        if self.waiting_queue:
            for req in self.waiting_queue:
                self._add_request(req)
                
    def _add_request(self,Mig_request:MigrateRequest)->None:
        request=self._convert_migrate_requests(Mig_request)
        self.running_queue.append(request)
        self.block_manager.allocate_prefilled_req(Mig_request)
        del Mig_request
        
    def _convert_migrate_requests(self, Mig_request:MigrateRequest)->Request:
        req=Mig_request.req
        self.block_manager.req_table[req.request_id]
        req.status=RequestStatus.RUNNING
        return req
    
    def abort_request(self, request:Request)->None:
        for (i,req) in enumerate(self.waiting_queue):
            if request.request_id==req.req.request_id:
                del self.waiting_queue[i]
                return
    
    def schedule_requests(self):
        batch=BatchedRequests()
        def _can_schedule(req:Request)->bool:
            if (len(batch.requests)<self.decode_scheduler_config.max_batch_size) and \
            (self.block_manager.can_append_slots(req)) and \
            (req.get_len()<self.decode_scheduler_config.max_token_num_each_req):
                return True
        for req in self.running_queue:
            if _can_schedule(req):
                batch.add_request(req)
        return batch
            
def get_FCFS_decode_scheduler(parallel_config:ParallelConfig,
                 decode_scheduler_config:DecodeSchedulerConfig,
                 cache_config:CacheConfig)->FCFS_DecodeStageScheduler:
    return FCFS_DecodeStageScheduler(parallel_config=parallel_config,
                                     decode_scheduler_config=decode_scheduler_config,
                                     cache_config=cache_config)
    
