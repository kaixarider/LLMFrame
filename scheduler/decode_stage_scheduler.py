from abc import ABC,abstractmethod
from MixFrame.request.request import Request,BatchedRequests,ScheduleType,MigrateRequests
from MixFrame.config.scheduler_config import DecodeSchedulerConfig,ParallelConfig
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
    def add_request(self,request:Request)->None:
        '''add_request to waiting queue'''
        raise NotImplementedError
    
    @abstractmethod
    def abort_request(self,request:Request)->None:
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
    def _can_schedule(self,request:Request)->None:
        '''determine whether this request can be scheduled'''
        raise NotImplementedError
    @abstractmethod
    def _convert_migrate_requests(self,Mig_request:MigrateRequests)->Request:
        raise NotImplementedError
class FCFS_DecodeStageScheduler(DecodeStageScheduler):
    def __init__(self,parallel_config:ParallelConfig,
                 decode_scheduler_config:DecodeSchedulerConfig):
        self.parallel_config=parallel_config
        self.decode_scheduler_config=decode_scheduler_config 
        
        #queues
        self.waiting_queue=[]
        self.running_queue=[]
        self.swap_queue=[]
    
    def add_request(self, Mig_request:MigrateRequests):
        request=self._convert_migrate_requests(Mig_request)
        self.running_queue.append(request)
    
    def _convert_migrate_requests(self, Mig_request:MigrateRequests)->Request:
        '''not fulfilled yet'''
