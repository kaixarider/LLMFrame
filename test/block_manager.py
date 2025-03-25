'''A blockmanager manages the allocation and free of blocks.It records the usage of memory logically.
When the number of blocks is not enough for requests,it swaps in some requests,so that the others can 
take usage of the blocks occupied by the swapped requests.
This file tests the function of blockmanager,including allocation,free,'''

from MixFrame.block.blockmanager import BlockManager
from MixFrame.request.request import Request
from MixFrame.request.sampling_parameter import SamplingParemeters
from MixFrame.tokenizer.tokenizer import  get_tokenizer,encode_tokens
import time
block_manager=BlockManager(block_size=16,num_gpu_blocks=30,num_cpu_blocks=30,watermark=0.01)
prompt="hello world,what's the whether like today?are you ok? nonono"
tokenizer=get_tokenizer('/home/wsy/workspace/llm/model/DeepSeek-R1-Distill-Qwen-7B')
token_ids=encode_tokens(tokenizer=tokenizer,text=prompt)
req=Request(arrival_time=time.time(),request_id=0,prompt=prompt,prompt_token_ids=token_ids,sampling_parameters=SamplingParemeters(n=1))
req2=Request(arrival_time=time.time(),request_id=1,prompt=prompt,prompt_token_ids=token_ids,sampling_parameters=SamplingParemeters(n=1))

'''test of allocation'''
block_manager.allocate(req)
block_manager.allocate(req2)
print(f"blocks used by req is {len(block_manager.req_table[req.request_id].used_blocks())}")
print(f"the rest num of blocks is {block_manager.gpu_block_allocator.get_num_free_block()}")

'''test of swap'''
block_manager.swap_out(req)
print(f"the rest num of blocks is {block_manager.gpu_block_allocator.get_num_free_block()} after swap out in gpu")
print(f"the rest num of blocks is {block_manager.cpu_block_allocator.get_num_free_block()} after swap out in cpu")
block_manager.swap_in(req)
print(f"the rest num of blocks is {block_manager.gpu_block_allocator.get_num_free_block()} after swap in")
print(f"the rest num of blocks is {block_manager.cpu_block_allocator.get_num_free_block()} after swap in in cpu")

'''test of free'''
block_manager.free(req)
print(f"the number of block after free req is {block_manager.gpu_block_allocator.get_num_free_block()}")
block_manager.free(req2)

print(f"the number of block after free req2 is {block_manager.gpu_block_allocator.get_num_free_block()}")
