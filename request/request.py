from typing import List,Optional
from MixFrame.request.sampling_parameter import SamplingParemeters
from MixFrame.config import ParallelConfig
from MixFrame.block.blocktable import Block
import logging
from enum import Enum
import enum
logger = logging.getLogger(__name__)
class ScheduleType(Enum):
    CB=1 #continuous batching
    PD=2 #prefill and decode disaggregation
class RequestStatus(Enum):
    WAITING=enum.auto() #requests to be prefilled
    RUNNING=enum.auto() #requests finished prefilling,decoding
    SWAPPED=enum.auto() #requests swapped in cpu
    
class Request:
    """A request contains the user's prompt, generated tokens and related information.
    Args:
        arrival_time: the absolute or relative time when the request arrives.
        request_id: the unique identifier for the request.
        prompt: the prompt provided by the user.
        prompt_token_ids: the token ids of the prompt.
        sampling_params: sampling parameters for the request.
        priority: the priority of this request, default is 0.
    """
    def __init__(
        self,
        arrival_time:float,
        request_id:int,
        prompt:str,
        prompt_token_ids:List[int],
        sampling_parameters:SamplingParemeters,
        priority:int =0,
    ):
        self.arrival_time=arrival_time
        #original_prompts
        self.request_id=request_id
        self.prompt=prompt
        self.prompt_token_ids=prompt_token_ids
        self.sampling_parameters=sampling_parameters
        self.priority=priority
        
        self.scheduled_time=None
        #generation
        self.generated_tokens=[]
        self.generated_token_ids=[]
        self.last_step_time=None
        self.finish_prefill_time=None
        self.is_finish=False
        self.is_running=False
        #status
        self.status:RequestStatus=RequestStatus.WAITING
        self._num_computed_tokens=0 ##prefilled tokens
        self._update_cached_all_tokens()
        
    def get_priority(self)->int:
        return self.priority
    
    def set_priority(self,priority:int)->None:
        self.priority=priority
        
    def set_scheduled_time(self,time:float)->None:
        self.scheduled_time=time
    def set_finish_prefill_time(self,time:float)->None:
        self.finish_prefill_time=time
    def get_output_len(self)->int:
        return len(self.generated_token_ids)
    
    def _check_stop_condition(self)->bool:
        if self.get_output_len()>=self.sampling_parameters.max_tokens:
            self.is_finish=True
            return True
        
        if not self.sampling_parameters.ignore_eos:
            if self.get_output_len() and \
                self.generated_token_ids[-1] in self.sampling_parameters.stop:
                self.is_finish=True
                return True
        return False
    
    def add_generated_tokens(self,token:str,token_id:int)->None:
        if self.get_output_len()>self.sampling_parameters.max_tokens:
            self.generated_token_ids=self.generated_token_ids[:self.sampling_parameters.max_tokens]
            self.generated_tokens=self.generated_tokens[:self.sampling_parameters.max_tokens]
            logger.log("Warn!The number of generated token is larger than \
                    max_tokens. The excess part is truncated.")
        self.generated_token_ids.append(token_id)
        self.generated_tokens.append(token)
        self._check_stop_condition()
    
    def is_prefill_stage(self)->bool:
        return (len(self.generated_token_ids)==0)
    
    def get_response(self)->str:
        return "".join(self.generated_tokens)
    
    def get_input_token_ids(self)->List[int]:
        if self.is_prefill_stage():
            return self.prompt_token_ids #prefill utilizes all tokens in prompt
        else:
            return [self.generated_token_ids[-1]] #decode utilizes only one 
    def get_len(self):
        return len(self.prompt_token_ids)+self.get_output_len() #output+input

    def _update_cached_all_tokens(self):
        self._cached_all_token_ids: List[int] = list(self.prompt_token_ids +
                                                     self.generated_token_ids)
class BatchedRequests:
    def __init__(self,
                 requests:Optional[List[Request]]=None,
                 schedule_type:Optional[ScheduleType]=None):
        if requests==None:
            self.requests=[]
        else:
            self.requests=requests
        if schedule_type==None:
            self._judge_scheduled_type()
        else:
            self.scheduled_type=schedule_type   
    def _judge_scheduled_type(self)->None:
        return
    
    def add_request(self,request:Request):
        self.requests.append(request)
        return
    
    def pop_finished_request(self)->List[Request]:
        finish_requests,unfinished_requests=[],[]
        for request in self.requests:
            if request._check_stop_condition():
                finish_requests.append(request)
            else:
                unfinished_requests.append(request)
        self.requests=finish_requests
        return finish_requests
    
    def schedule_type(self)->ScheduleType:
        return self.scheduled_type

class MigrateRequests:
    def __init__(self,req:Request,
                 para_config:ParallelConfig)->None:
        
        self.req=req
        self.para_config=para_config
        self.blocks:List[List[int]]=[]

    def add_block_token_ids(self,block_token_ids:List[int])->None:
        self.blocks.extend(block_token_ids)
        return 
        