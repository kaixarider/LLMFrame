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
        
def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


def get_gpu_memory_usage(gpu: int = 0):
    """
    Python equivalent of nvidia-smi, copied from https://stackoverflow.com/a/67722676
    and verified as being equivalent ✅
    """
    output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"

    try:
        memory_use_info = output_to_list(
            sp.check_output(COMMAND.split(), stderr=sp.STDOUT)
        )[1:]

    except sp.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )

    return int(memory_use_info[gpu].split()[0])


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
class ScheduleType(enum.Enum):
    PD=enum.auto()
    CB=enum.auto()

class InferenceStage(enum.Enum):
    prefill=enum.auto()
    decode=enum.auto()