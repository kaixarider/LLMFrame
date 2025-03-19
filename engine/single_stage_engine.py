'''the basic type of engine.Engine is based on scheduler and schedule requests according to
schedule policy'''
from abc import ABC,abstractmethod
from typing import Tuple,List,Callable,Optional
import asyncio
import enum
import logging
import torch
from MixFrame.config import ParallelConfig,DisParallelConfig,ModelConfig,CacheConfig
from MixFrame.config import DecodeSchedulerConfig,PrefillSchedulerConfig,SchedulerConfig
from MixFrame.scheduler.decode_stage_scheduler import DecodeStageScheduler,get_FCFS_decode_scheduler
from MixFrame.scheduler.prefill_stage_scheduler import PrefillStageScheduler, get_FCFS_prefill_scheduler
from MixFrame.request.request import Request,MigrateRequest,BatchedRequests
from MixFrame.worker.worker import Worker
from MixFrame.tokenizer.tokenizer import get_tokenizer
from MixFrame.util import SchedulerType,InferenceStage
from MixFrame.block.blockmanager import BlockManager
logger= logging.getLogger(__name__)
class StepOutput:
    '''It contains the information a request generates in one step'''
    def __init__(self,req:Request,new_token_id:int,new_token:str):
        self.request = req
        self.request_id = req.request_id
        self.prompt = req.prompt
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finish = req.is_finish
        
    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.request_id}, "
            f"new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}, "
            f"is_finished={self.is_finish})"
        )
        
class SingleStepEngine(ABC):
    @abstractmethod
    def get_scheduler(self,type:SchedulerType)->(DecodeStageScheduler|PrefillStageScheduler):
        raise NotImplementedError
    
 
    def __init__(
        self,
        stage: InferenceStage,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sche_config:SchedulerConfig,
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],   # The LLMEngine's callback function when a new StepOutput of a particular request is generated
    )->None:
        self.stage = stage
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.sche_config = sche_config
        self.engine_on_new_step_output_callback = engine_on_new_step_output_callback

        self.tokenizer = get_tokenizer(
            tokenizer_name=model_config.tokenizer
        )
        
        # workers[i][j] is the j-th tensor-parallel worker in pipeline stage i
        self.workers:List[List[Worker]] = []
    
    async def initialize(self):
        logger.info(f"Initializing {self.stage.name} workers")
        await self._init_workers()
        
        logger.info(f"Initializing {self.stage.name} models")
        await self._init_model()
        
        logger.info(f"Initializing {self.stage.name} kvcaches")
        self.num_gpu_blocks, self.num_cpu_blocks = await self._init_kvcache()
        
        self.scheduler = self.get_scheduler()

        logger.info(f"Scheduler: {self.scheduler}")

    def _remote_call_all_workers_async(self, func_name: str, *args):
        """
        call func_name asynchronously on all workers, return the futures immediately
        """
        handlers = []
        for stage in self.workers:
            for worker in stage:
                handlers.append(getattr(worker, func_name).remote(*args))
        return handlers
    
    async def _init_model(self):
        return 
    async def _init_workers(self):
        for i in range(self.parallel_config.pp_size):
            workers=[]
            for j in range(self.parallel_config.tp_size):
                tmp_para_config=self.parallel_config
                tmp_para_config.pp_rank=i
                tmp_para_config.tp_rank=j
                workers.append(Worker(gpu_id=torch.cuda.current_device(),stage=self.stage,
                                      parallel_config=tmp_para_config,cache_config=self.cache_config,
                                      sche_config=self.sche_config,model_config=self.model_config))
            self.workers.append(workers)
            
    async def _init_kvcache(self):
        return
class PrefillEngine(SingleStepEngine):
    def __init__(
        self,
        stage: InferenceStage,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sche_config:SchedulerConfig,
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],   # The LLMEngine's callback function when a new StepOutput of a particular request is generated
    )->None:
        super().__init__(stage=stage,model_config=model_config,
                         parallel_config=parallel_config,cache_config=cache_config,
                         sche_config=sche_config,engine_on_new_step_output_callback=engine_on_new_step_output_callback)

    def get_scheduler(self,type:SchedulerType)->None:
        match type:
            case SchedulerType.FCFS:
                self.scheduler=get_FCFS_prefill_scheduler(sche_config=self.sche_config,parallel_config=self.parallel_config,cache_config=self.cache_config)
            case _:
                raise TypeError("There is no such schedule type")
    
    def initialize(self):
        return super().initialize()
    
class DecodeEngine(SingleStepEngine):
    def __init__(
        self,
        stage: InferenceStage,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sche_config:SchedulerConfig,
        engine_on_new_step_output_callback: Callable[[int, StepOutput], None],   # The LLMEngine's callback function when a new StepOutput of a particular request is generated
    ):
        super().__init__(stage=stage,model_config=model_config,
                         parallel_config=parallel_config,cache_config=cache_config,
                         sche_config=sche_config,engine_on_new_step_output_callback=engine_on_new_step_output_callback)
    
    def getscheduler(self,type:SchedulerType)->None:
        match type:
            case SchedulerType.FCFS:
                return get_FCFS_decode_scheduler(sche_config=self.sche_config,parallel_config=self.para_config,cache_config=self.cache_config)
            case _:
                raise TypeError(f"There is no such {type} schedule type!")

    def initialize(self):
        return super().initialize()
    
class CoTEngine(SingleStepEngine):
    def __init__(self,CoT_sche_config,para_config,model_config)->None:
        return
    
class ResponseEngine(SingleStepEngine):
    def __init__(self,Response_sche_config,para_config,model_config)->None:
        return
