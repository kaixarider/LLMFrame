class ParallelConfig:
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
    