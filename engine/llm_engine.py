'''create engine based on scheduler,
schedule requests according to schedule policy,assign requests to workers.
Consists of PrefillEngine and DecodeEngine'''
import asyncio
from MixFrame.tokenizer.tokenizer import get_tokenizer
from MixFrame.scheduler.decode_stage_scheduler import DecodeSchedulerConfig,get_decode_scheduler
from MixFrame.scheduler.prefill_stage_scheduler import PrefillSchedulerConfig,get_prefill_scheduler
from MixFrame.config import PrefillSchedulerConfig,DecodeSchedulerConfig,DisParallelConfig,ModelConfig,CacheConfig
from MixFrame.engine.single_stage_engine import PrefillEngine,DecodeEngine
from MixFrame.util import InferenceStage
class LLMEngine():
    def __init__(self,
                 cache_config:CacheConfig,
                 model_config:ModelConfig,
                 dis_parallel_config:DisParallelConfig,
                 prefill_scheduler_config:PrefillSchedulerConfig,
                 decode_scheduler_config:DecodeSchedulerConfig):
        self.cache_config=cache_config
        self.model_config=model_config
        self.dis_parallel_config=dis_parallel_config
        self.prefill_scheduler_config=prefill_scheduler_config
        self.decode_scheduler_config=decode_scheduler_config
        
        self.tokenizer=get_tokenizer(model_config.tokenizer)
        self.prefill_scheduler=get_prefill_scheduler(sche_config=prefill_scheduler_config,parallel_config=self.dis_parallel_config.prefill_parallel_config,
                                                     cache_config=self.cache_config)
        self.prefill_engine=PrefillEngine(stage=InferenceStage.prefill,model_config=self.model_config,
                                          parallel_config=self.dis_parallel_config.prefill_parallel_config,
                                          cache_config=self.cache_config ,sche_config=self.prefill_scheduler_config)
        self.bridge_queue=asyncio.Queue()
        self.decode_scheduler=get_decode_scheduler(parallel_config=self.dis_parallel_config.decode_parallel_config,
                                                   decode_scheduler_config=decode_scheduler_config,cache_config=self.cache_config)
        self.decode_engine=DecodeEngine(stage=InferenceStage.decode,model_config=model_config,
                                        parallel_config=self.dis_parallel_config.decode_parallel_config,
                                        cache_config=cache_config,sche_config=decode_scheduler_config)
        
    def step():
        '''contains three steps.
            -Schedule. Scheduler receives prompts and select some or all of them for inference.
            -Inference. Worker receives selected prompts and start inference.Once prefill stage finishing,
            scheduler receive the results and decide whether the batch should be sent for decode,or continuous
            batching locally. After the type is determined,decode stage will start.
            -Process outputs. Decoder will handle the outputs.'''
        