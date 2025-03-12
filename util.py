from typing import List
def chunk_list(lst:List,size:int):
    for i in range(0,len(lst),size):
        yield lst[i:i+size]
        
        