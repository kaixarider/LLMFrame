from MixFrame.scheduler.prefill_stage_scheduler import get_prefill_scheduler
from MixFrame.scheduler.decode_stage_scheduler import get_decode_scheduler
from MixFrame.config import ParallelConfig,PrefillSchedulerConfig,DecodeSchedulerConfig,DisParallelConfig,CacheConfig
from MixFrame.tokenizer.tokenizer import get_tokenizer,encode_tokens
from MixFrame.request.request import Request
from MixFrame.request.request import SamplingParemeters

import time
prefill_para_config=ParallelConfig()
decode_para_config=ParallelConfig()
prefill_sche_config=PrefillSchedulerConfig(parallel_config=prefill_para_config,
                                           policy='fcfs',max_batch_size=10,max_token_num_each_req=8192)
decode_sche_config=DecodeSchedulerConfig(parallel_config=decode_para_config,policy='fcfs',
                                         max_batch_size=10,max_token_num_each_req=8192)
cache_config=CacheConfig(block_size=16,max_num_blocks_per_req=8192/16,
                         gpu_memory_utilization=0.9)
prefill_schduler=get_prefill_scheduler(sche_config=prefill_sche_config,
                                       parallel_config=prefill_para_config,cache_config=cache_config,
                                       num_gpu_blocks=2,num_cpu_blocks=20)
decode_scheduler=get_decode_scheduler(parallel_config=decode_para_config,
                                      decode_scheduler_config=decode_sche_config,cache_config=cache_config,
                                    num_gpu_blocks=2,num_cpu_blocks=20)
prompt="hello world,what's the whether like today?are you ok? nonono.where there is a will"
tokenizer=get_tokenizer('/home/wsy/workspace/llm/model/DeepSeek-R1-Distill-Qwen-7B')
token_ids=encode_tokens(tokenizer=tokenizer,text=prompt)
req=Request(arrival_time=time.time(),request_id=0,prompt=prompt,prompt_token_ids=token_ids,sampling_parameters=SamplingParemeters(n=1))
req2=Request(arrival_time=time.time(),request_id=1,prompt=prompt,prompt_token_ids=token_ids,sampling_parameters=SamplingParemeters(n=1))
prefill_schduler.add_request(req)
prefill_schduler.add_request(req2)
batch=prefill_schduler.select_requests()
print(len(batch.requests))
print(prefill_schduler.block_manager.gpu_block_allocator.get_num_free_block())
prefill_schduler.block_manager.free(req)

batch=prefill_schduler.select_requests()
print(len(batch.requests))
print(prefill_schduler.block_manager.gpu_block_allocator.get_num_free_block())