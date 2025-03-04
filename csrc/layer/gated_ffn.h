#pragma once

#include "../kernel/addbias.h"
#include "../kernel/activations.cuh"
#include "../util/nccl_utils.h"
namespace layer{
    template<typename T>
    void gated_ffn{
        T*  output,
        T* __restrict__ input,//[token num,hidden size]
        T* __restrict__ fc1_weight,//[hidden size,inter_size/tp_size]
        T* __restrict__ fc1_bias,
        T* __restrict__ 
    }
}