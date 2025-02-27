#pragma once
#include "util/cuda_utils.h"
namespace kernel{

template<typename T>
void addbias(
    T* output,
    const T* input,
    const T* bias,
    const int64_t size
);

template<typename T>
void addbiasBatched(
    T* output,
    const T* input,
    const T* bias,
    const int64_t batch_size,
    const int64_t size
);
}