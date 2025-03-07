'''create engine based on scheduler,
schedule requests according to schedule policy,assign requests to workers.
Consists of PrefillEngine and DecodeEngine'''
from MixFrame.util.tokenizer import get_tokenizer
from MixFrame.scheduler.decode_stage_scheduler import DecodeSchedulerConfig,FCFS_DecodeStageScheduler
from MixFrame.scheduler.prefill_stage_scheduler import PrefillSchedulerConfig,FCFS_PrefillStageScheduler
class LLMEngine():
    def step():
        '''contains three steps.
            -Schedule. Scheduler receives prompts and select some or all of them for inference.
            -Inference. Worker receives selected prompts and start inference.Once prefill stage finishing,
            scheduler receive the results and decide whether the batch should be sent for decode,or continuous
            batching locally. After the type is determined,decode stage will start.
            -Process outputs. Decoder will handle the outputs.'''
        