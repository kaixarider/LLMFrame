from MixFrame.block.blockmanager import BlockManager
from MixFrame.request.request import Request
from MixFrame.request.sampling_parameter import SamplingParemeters
from MixFrame.tokenizer.tokenizer import  get_tokenizer,decode_tokens,encode_tokens
import time
block_manager=BlockManager(block_size=16,num_gpu_blocks=30,num_cpu_blocks=10,watermark=0.01)
prompt="hello world,what's the whether like today?are you ok? nonono"
tokenizer=get_tokenizer('/home/wsy/workspace/llm/model/DeepSeek-R1-Distill-Qwen-7B')
token_ids=encode_tokens(tokenizer=tokenizer,text=prompt)
req=Request(arrival_time=time.time(),request_id=0,prompt=prompt,prompt_token_ids=token_ids,sampling_parameters=SamplingParemeters(n=1))
req2=Request(arrival_time=time.time(),request_id=1,prompt=prompt,prompt_token_ids=token_ids,sampling_parameters=SamplingParemeters(n=1))
block_manager.allocate(req)
block_manager.allocate(req2)
print(block_manager.req_table[req.request_id].used_blocks()[0].block_id)
print(block_manager.gpu_block_allocator.get_num_free_block())
block_manager.free(req)
print(block_manager.gpu_block_allocator.get_num_free_block())
prompt2="hello world,what's the whether like today?are you ok? nonono thief \
    where there is a will there is a way.May the force be with you.wish you good luck."
token_ids2=encode_tokens(tokenizer=tokenizer,text=prompt2)
req=Request(arrival_time=time.time(),request_id=2,prompt=prompt2,prompt_token_ids=token_ids2,sampling_parameters=SamplingParemeters(n=1))
block_manager.allocate(req)
for i in block_manager.req_table[req.request_id].used_blocks():
    print(f"used block id is {i._token_ids}")
block_manager.free(req2)
