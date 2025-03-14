from abc import ABC,abstractmethod
from typing import List
from enum import Enum,auto
from MixFrame.request.request import Request,BatchedRequests,ScheduleType,MigrateRequests
from MixFrame.config import PrefillSchedulerConfig,ParallelConfig,CacheConfig
from MixFrame.block.blockmanager import BlockManager
class BatchingMethod:
    CB=auto()#continuous batching
    PD=auto()#pd disaggregation
    
class PrefillStageScheduler(ABC):
    '''Prefill stage scheduelr schedules requests to prefill,
    then it determines to decode locally,or migrate'''
    @abstractmethod
    def __init__(self,
                 parallel_config:ParallelConfig,
                 prefill_scheduler_config:PrefillSchedulerConfig):
        '''initiate prefill stage scheduler'''
        raise NotImplementedError
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
    @abstractmethod
    def CB_or_PD(self,BatchedRequests:BatchedRequests)->ScheduleType:
        '''determine whether continuous batching(CB) or Prefill Decode Disaggregation(PD)
        suit a batch'''
        raise NotImplementedError
    @abstractmethod
    def _can_schedule(self,request:Request):
        '''determine whether this request can be scheduled'''
        raise NotImplementedError
    @abstractmethod
    def _convert_request_to_Migrequest(self,request:Request):
        raise NotImplementedError
class FCFS_PrefillStageScheduler(PrefillStageScheduler):
    def __init__(self,
                 parallel_config:ParallelConfig,
                 prefill_scheduler_config:PrefillSchedulerConfig,
                 cache_config:CacheConfig)->None:
        assert prefill_scheduler_config.policy=='fcfs',"FCFS scheduler should be served for \
            fcfs policy!"
        self.parallel_config=parallel_config
        self.prefill_scheduler_config=prefill_scheduler_config
        self.block_manager=BlockManager(block_size=cache_config.block_size,num_gpu_blocks=cache_config.num_gpu_blocks,
                                        num_cpu_blocks=cache_config.num_cpu_blocks)# to be finished
        '''
        four queues.
        -waiting queue:requests'''
        self.waiting_queue:List[Request]=[]
        self.running_queue=[]
        self.migrate_queue=[]
        self.swap_queue=[]
    
    def add_request(self, request:Request)->None:
        self.waiting_queue.append(request)
        self.block_manager.allocate(request)
        
    def abort_request(self, request_id:int)->None:
        for (i,request) in enumerate(self.waiting_queue):
            if request_id==request.request_id:
                del self.waiting_queue[i]
                return

    def select_requests(self,batching_method:BatchingMethod)->BatchedRequests:
        batch=BatchedRequests()
        if batching_method==BatchingMethod.CB:
            for request in self.running_queue:
                ##缺少continous batching的部分
            return batch
        elif batching_method==BatchingMethod.PD:
            for request in self.waiting_queue:
                '''There are several limitations '''
                if len(batch.requests)<self.prefill_scheduler_config.max_batch_size and self.block_manager.can_allocate(request):
                    batch.add_request(request)
            return batch
        else:
            raise ValueError(f"Error!There is no {batching_method} batching method")
    
    def migrate_requests(self,batch:BatchedRequests,target:int)->None:
        assert batch.schedule_type()==ScheduleType.PD,"continuous batching doesn't need to migrate!"
        for req in batch.requests:
            block_table=self.block_manager.req_table[req.request_id]
            ##缺少block_table中读取block的函数，用于迁移
            miragate_request=MigrateRequests(req=req,para_config=self.parallel_config)
            blocks=block_table.used_blocks()
            miragate_request.add_blocks(blocks)