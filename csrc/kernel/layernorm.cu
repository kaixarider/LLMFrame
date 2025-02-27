#include "../util/reduction.cuh"    
namespace kernel{
    template<typename T>
    __global__ void layernorm_kernel(
        const __restrict__ T* input,//[...,hidden_size]
        const __restrict__ T* output,//[...,hidden_size]
        const __restrict__ T* weight,//[hiddensize]
        const float epsilon, const int num_tokens,
        const int hidden_size
    )
    {
        __shared__ float s_variance;
        float local_variance;
        for(int i=threadIdx.x;i<hidden_size;i+=blockDim.x)
        {
            const float x = (float)input[blockIdx.x * hidden_size + i];
            local_variance+=x*x;
        }
        local_variance=block_sum_value(local_variance);
        if (threadIdx.x == 0) {
            s_variance = rsqrtf(local_variance / hidden_size + epsilon);
          }
          __syncthreads();
          for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
            float x = (float)input[blockIdx.x * hidden_size + idx];
            out[blockIdx.x * hidden_size + idx] =
                ((T)(x * s_variance)) * weight[idx];
          }
    }
    template<typename T,int width>
    __global__ std::enable_if_t<(width>0)>
    fused_add_rms_norm_kernel(
        
    )
    {

    }
}