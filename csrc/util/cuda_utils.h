#pragma once

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define INDEX_3D(dim3,idx3,dim2,idx2,dim1,idx1)=\
((idx3)*(dim2)*(dim1)+(idx2)*(dim1)+(idx1)) //index the value in 3D-vector
inline int64_t cuda_memory_size() {
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    return total_byte - free_byte;
}