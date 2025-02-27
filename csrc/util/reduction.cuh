#pragma once
#define FINAL_MASK 0xffffffffu
#include "../util/calculate.h"
#include <cub/cub.cuh>
template <typename T>
__inline__ __device__ T warp_max_value(T val)//find max value in a warp
{
    using WarpReduce = cub::WarpReduce<T>;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    // 通过 WarpReduce 归约
    float warp_max = WarpReduce(temp_storage).Reduce(val, cub::Max{});

    return warp_max;
}
 
template<typename T>
__inline__ __device__ T block_max_value(T val)//find max value in a block
{
    using BlockReduce = cub::BlockReduce<T, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    float block_max = BlockReduce(reduceStore).Reduce(val, cub::Max{});

    return block_max;
}

template<typename T>
__inline__ __device__ T block_sum_value(T val)
{
    using BlockReduce = cub::BlockReduce<T, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    float block_sum = BlockReduce(reduceStore).Reduce(val, cub::Sum{});

    return block_sum;
}
