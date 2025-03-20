'''the basic type of engine.Engine is based on scheduler and schedule requests according to
schedule policy'''
from abc import ABC,abstractmethod
from typing import Tuple,List,Callable,Optional
import asyncio
import enum
import logging
import torch
import torch.distributed as dist 
import torch.distributed.rpc as rpc
from MixFrame.config import ParallelConfig,DisParallelConfig,ModelConfig,CacheConfig
from MixFrame.config import DecodeSchedulerConfig,PrefillSchedulerConfig,SchedulerConfig
from MixFrame.scheduler.decode_stage_scheduler import DecodeStageScheduler,get_decode_scheduler
from MixFrame.scheduler.prefill_stage_scheduler import PrefillStageScheduler, get_prefill_scheduler
from MixFrame.request.request import Request,MigrateRequest,BatchedRequests
from MixFrame.worker.worker import Worker
from MixFrame.tokenizer.tokenizer import get_tokenizer
from MixFrame.util import SchedulerType,InferenceStage
from MixFrame.block.blockmanager import BlockManager
logger= logging.getLogger(__name__)
SLEEP_WHEN_CONTEXT_NO_REQUEST=0.3
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
         # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline: List[BatchedRequests] = []
        self.batches_ret_futures = []
    async def initialize(self):
        logger.info(f"Initializing {self.stage.name} workers")
        await self._init_workers()
        
        logger.info(f"Initializing {self.stage.name} models")
        await self._init_model()
        
        logger.info(f"Initializing {self.stage.name} kvcaches")
        self.num_gpu_blocks, self.num_cpu_blocks = await self._init_kvcache()
        
        self.scheduler = self.get_scheduler()

        logger.info(f"Scheduler: {self.scheduler}")


    
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
    

    @abstractmethod
    async def _step(self)->None:
        raise NotImplementedError()
    
    
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
        self.scheduler=get_prefill_scheduler(sche_config=self.sche_config,parallel_config=self.parallel_config,cache_config=self.cache_config)
    

    def initialize(self):
        return super().initialize()
    
    def _clear_req_resource(self,req:Request):
        self.scheduler.clear_req(req)
        PrefillEngine.remote_call_all_workers_async("clear_req",req)
    def _clear_batch_resource(self,batch:BatchedRequests):
        for req in batch:
            self._clear_req_resource(req)

    @staticmethod
    def remote_call_all_workers_async(func_name: str, *args):
        """
        在所有 worker 进程上异步调用 func_name，返回一个 Future 列表
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        futures = []
        for dst_rank in range(world_size):
            if dst_rank != rank:  # 避免 self 调用自己
                fut = rpc.rpc_async(f"worker{dst_rank}", getattr(Worker(), func_name), args=args)
                futures.append(fut)
        return futures  # 返回所有 Future 句柄
    
    async def _step(self)->None:
        batch=self.scheduler.select_requests()
        '''select requests for batching'''
        if len(batch) == 0:
            # Two cases may cause len(batched_requests) == 0:
            # 1. No request in the waiting queue
            # 2. No enough free blocks (e.g. the decoding stage is too slow)
            self.batches_in_pipeline.append(batch)
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_CONTEXT_NO_REQUEST)
        else:
            logger.info(f"(context) Forwarding with lengths {[len(request.prompt_token_ids) for request in batch.requests]}")
            # allocate blocks as needed
            for req in batch:
                self.scheduler.block_manager.allocate(req)
                
            self.batches_in_pipeline.append(batch)
            futures=self.remote_call_all_workers_async(
                "step",
                
            )
            pp_size = self.parallel_config.pp_size
            tp_size = self.parallel_config.tp_size
                # only the leader of the last stage return valid output, i.e., generated tokens ids
            self.batches_ret_futures.append(futures[(pp_size - 1) * tp_size])
            
        if len(self.batches_in_pipeline) == self.parallel_config.pp_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
                # No request in the batch
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                generated_tokens_ids = await self.batches_ret_futures[0]
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

        self.scheduler=get_decode_scheduler(sche_config=self.sche_config,parallel_config=self.parallel_config,cache_config=self.cache_config)


    def initialize(self):
        return super().initialize()
    def _clear_req_resource(self,req:Request):
        self.scheduler.clear_req(req)
        PrefillEngine.remote_call_all_workers_async("clear_req",req)
        
    def _clear_batch_resource(self,batch:BatchedRequests):
        for req in batch:
            self._clear_req_resource(req)
    
    @staticmethod
    def remote_call_all_workers_async(func_name: str, *args):
        """
        在所有 worker 进程上异步调用 func_name，返回一个 Future 列表
        """
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        futures = []
        for dst_rank in range(world_size):
            if dst_rank != rank:  # 避免 self 调用自己
                fut = rpc.rpc_async(f"worker{dst_rank}", getattr(Worker(), func_name), args=args)
                futures.append(fut)
        return futures  # 返回所有 Future 句柄
class CoTEngine(SingleStepEngine):
    def __init__(self,CoT_sche_config,para_config,model_config)->None:
        return
    
class ResponseEngine(SingleStepEngine):
    def __init__(self,Response_sche_config,para_config,model_config)->None:
        return
