from abc import ABC,abstractmethod
from MixFrame.request.request import Request,BatchedRequests,ScheduleType
from MixFrame.config import PrefillSchedulerConfig,ParallelConfig
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
    def _can_schedule(self,request:Request):
        '''determine whether this request can be scheduled'''
        raise NotImplementedError
    @abstractmethod
    def _convert_request_to_Migrequest(self,request:Request):
        raise NotImplementedError
class FCFS_PrefillStageScheduler(PrefillStageScheduler):
    def __init__(self,
                 parallel_config:ParallelConfig,
                 prefill_scheduler_config:PrefillSchedulerConfig):
        assert prefill_scheduler_config.policy=='fcfs',"FCFS scheduler should be served for \
            fcfs policy!"
        self.parallel_config=parallel_config
        self.prefill_scheduler_config=prefill_scheduler_config
        
        '''
        four queues.
        -waiting queue:requests'''
        self.waiting_queue=[]
        self.running_queue=[]
        self.migrate_queue=[]
        self.swap_queue=[]
    
    def add_request(self, request:Request):
        self.waiting_queue.append(request)
    
    