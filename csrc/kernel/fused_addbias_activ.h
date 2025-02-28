#pragma once
#include "activations.cuh"
#include <cstdint>


namespace kernel{
template<typename T>
void fused_activation_multiply(
    T* __restrict__ out,
    const T* __restrict__ A,
    const T* __restrict__ B,
    int n,
    ActivationType activationType
)
}