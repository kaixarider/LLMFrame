from abc import ABC,abstractmethod
from enum import Enum


from MixFrame.request.request import BatchedRequests,Request
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
    def __init__(self,
                 parallel_config:ParallelConfig):
        self.tp_size=parallel_config.tp_size
        self.dp_size=parallel_config.dp_size
        self.waiting_queue=[]
        
    def add_request(self, request:Request)->None:
        self.waiting_queue.append(request)
        
    def abort_request(self, request_id:int):
        for (i,request) in enumerate(self.waiting_queue):
            if request.request_id==request_id:
                del self.waiting_queue[i]
                return
    
    def get_next_batch_and_pop(self):
        '''condition judge and add request'''
