from typing import List
import torch
import psutil
import subprocess as sp
import enum
import random
import numpy as np
GB=2**30
MB=2**20
def chunk_list(lst:List,size:int):
    for i in range(0,len(lst),size):
        yield lst[i:i+size]
        
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
class BatchingType(enum.Enum):
    PD=enum.auto()
    CB=enum.auto()

class InferenceStage(enum.Enum):
    prefill=enum.auto()
    decode=enum.auto()

class SchedulerType(enum.Enum):
    FCFS=enum.auto()