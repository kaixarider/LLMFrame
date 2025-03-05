from abc import ABC,abstractmethod
from enum import Enum


from MixFrame.request.request import BatchedRequests,Request,ScheduleType,MigrateRequests
from MixFrame.config.parallel_config import ParallelConfig
class SchedulerConfig(ABC):
    @abstractmethod
    def add_request(self,request:Request)->None:
        raise NotImplementedError() #add_request to the waiting queue
    
    @abstractmethod
    def abort_request(self,request_id:int)->None:
        raise NotImplementedError() #abort unqualified request
    
    @abstractmethod
    def get_next_batch_and_pop(self) -> BatchedRequests:
        """
        Get a batch of requests for the execution of next iteration and
        pop the requests in the batch from the waiting queue.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_waiting_requests(self) -> int:
        """
        Get the number of requests that are waiting for processing.
        """
        raise NotImplementedError()

    @abstractmethod
    def print_status(self) -> None:
        """
        Print the status of the scheduler.
        """
        raise NotImplementedError()

class PrefillScheduleConfig(SchedulerConfig):
    '''It performs prefill stage of requests.Then it performs '''
    '''There are four queues,waiting queue,running queue,migrate_queue and swapped queue'''
    def __init__(self,
                 parallel_config:ParallelConfig):
        self.tp_size=parallel_config.tp_size
        self.dp_size=parallel_config.dp_size
        
        self.waiting_queue=[]
        self.migrate_queue=[]
        self.running_queue=[]
        self.swapped_queue=[]
    def add_request(self, request:Request)->None:
        self.waiting_queue.append(request)
        
    def abort_request(self, request_id:int):
        for (i,request) in enumerate(self.waiting_queue):
            if request.request_id==request_id:
                del self.waiting_queue[i]
                return
   
    def get_next_batch_and_pop(self) -> BatchedRequests:
        '''continuous batching or pure prefill,select from waiting queue or running queue'''
        

    def get_num_waiting_requests(self) -> int:
        """
        Get the number of requests that are waiting for processing.
        """
    def print_status(self) -> None:
        '''print status'''
        
    def _alloc(self,request:Request)->bool:
        '''try to alloc for request,if possible,it can be scheduled,else wait'''
    def _can_schedule(self)->None:
        ''''''

    def check_CB_or_PD(self,Batchedrequests:BatchedRequests)->ScheduleType:
        return Batchedrequests.scheduled_type
    
    def migrate_batch(self,Batchedrequests:BatchedRequests)->None:
        '''to be finished'''
        
    def schedule_decode(self)->None:
        '''schedule from running queue'''
    
class DecodeScheduleConfig(SchedulerConfig):
    def __init__(self,
                 parallel_config:ParallelConfig):
        self.tp_size=parallel_config.tp_size
        self.dp_size=parallel_config.dp_size
        
        self.waiting_queue=[]
        self.swapped_queue=[]
    
    def add_request(self, request:MigrateRequests):
        '''add request to waiting queue and transform MigrateRequest to Request'''
        
    def abort_request(self, request_id):
        '''abort requests that can not generate'''
        
    def get_next_batch_and_pop(self) -> BatchedRequests:
        '''select requests and inference'''
        
    def get_num_waiting_requests(self):
        '''get number of waiting requests'''