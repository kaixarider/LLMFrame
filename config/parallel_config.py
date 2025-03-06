
from typing import Dict
class ParallelConfig:
    '''config numbers of parallel nodes'''
    def __init__(
      self,
      tp_size:int=1,#tensor_parallel
      tp_rank:int=0,
      dp_size:int=1,#data_parallel
      dp_rank:int=0
    )->None:
        self.tp_size=tp_size
        self.tp_rank=tp_rank
        self.dp_size=dp_size
        self.dp_rank=dp_rank
        
        self.world_size=dp_size*tp_size
        self.use_parallel=self.world_size>1
    
    def show_config(self)->Dict:
      return {"tp_size":self.tp_size,
              "tp_rank":self.tp_rank,
              "dp_size":self.dp_size,
              "dp_rank":self.dp_size}
    
class DisParallelConfig:
    def __init__(
      self,
      PrefillParallelConfig:ParallelConfig,
      DecodeParallelConfig:ParallelConfig
    )->None:
      self.prefill_parallel_config=PrefillParallelConfig
      self.decode_parallel_config=DecodeParallelConfig
      
    def num_workers(self):
      return self.decode_parallel_config.world_size+self.prefill_parallel_config.world_size
    
  
      