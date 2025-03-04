from abc import ABC,abstractmethod

class Scheduler_Config(ABC):
    @abstractmethod
    def add_request(self,request)->None:
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