#include "fused_addbias_activ.h"

namespace kernel{
    template<typename T,ActivationType acttype>
    __global__ void fused_addbias_batched_activation_kernel(
        T* __restrict__ output,
        const T* __restrict__ input,
        const T* __restrict__ bias,
        const int batch,
        const int size
    )
    {
        typedef std::conditional_t<std::is_same_v<T, half>, half2, float2> T2;
	    #pragma unroll
        for(int i=threadIdx.x+blockIdx.x*blockDim.x;i<size/2;i+=blockDim.x*gridDim.x)
        {
            const int s=i%(size/2);
            T2 input_elem=((const T2*)input)[i];
            T2 bias_elem = ((const T2*)bias)[s];
            T2 output_elem = {
                applyActivation<T, ACTIVATION_TYPE>(input_elem.x + bias_elem.x),
                applyActivation<T, ACTIVATION_TYPE>(input_elem.y + bias_elem.y)
            };
            ((T2*)output)[i] = output_elem;
        }
    }

    template<typename T,ActivationType acttype>
    void fused_addbias_batched_activation(
        T* __restrict__ output,
        const T* __restrict__ input,
        const T* __restrict__ bias,
        const int batch,
        const int size,
        ActivationType acttype
    )
    {
        assert_whenever(size%2==0);
        const int block_size=256;
        const int grid_size=(size*batch/2 + block_size - 1) / block_size;
        switch(acttype){
            case ActivationType.RELU:{
                fusedAddbiasBatchedActivationKernel<T, ActivationType::RELU><<<gridSize, blockSize>>>(output, input, bias, batch, size);
                break;
            }
            case ActivationType.SILU:{
                fusedAddbiasBatchedActivationKernel<T, ActivationType::SILU><<<gridSize, blockSize>>>(output, input, bias, batch, size);
                break;
            }
            case ActivationType.GELU:
            {
                fusedAddbiasBatchedActivationKernel<T, ActivationType::GELU><<<gridSize, blockSize>>>(output, input, bias, batch, size);
                break;
            }
            default:{
                assert(false)
            }
        }

    }
    template void fusedAddbiasBatchedActivation(half* output, const half* input, const half* bias, const int64_t batch, const int64_t size, ActivationType activation_type);
    template void fusedAddbiasBatchedActivation(float* output, const float* input, const float* bias, const int64_t batch, const int64_t size, ActivationType activation_type);
}