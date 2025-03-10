
from MixFrame.config import CacheConfig,ModelConfig,ParallelConfig
from MixFrame.request.request import Request
from block_allocator import BlockLocation,BlockList
from block_table import BlockTable,BlockAllocator
from enum import Enum
from typing import Callable,List

class ManagerType(Enum):
    '''type of blockmanage
    -type one generates kv cache ant utilize it(if CB is taken)
    -type two takes migration and executes decode'''
    TypeOne=1,
    TypeTwo=2
class BlockManager:
    def __init__(
        self,
        
    )
    
   