
#include "softmax.cuh"
#include "../util/reduction.cuh"
#include "../util/cuda_utils.h"
namespace kernel{
    /*calculate softmax of a sequence[num head,seq_length]*/
    template<typename T>
    __global__ void scale_softmax_kernel(
        const T* input,//[head_dim,seq_len] 
        T* __restrict__  output,
        float scale,
        int64_t input_length
    )
    {
        __shared__ float s_max=0,s_sum=0;
        uint64_t start_idx=input_length*blockIdx.x+threadIdx.x;//griddim=numhead
        uint64_t end_idx=input_length*blockIdx.x+input_length;
        float local_max=-1e20f;
        #pragma unroll
        for(uint64_t i=start_idx;i<end_idx;i+=blockDim.x){
            float value=input[i];
            local_max=MAX(local_max,value/scale);
        }
        float max_val = blockDim.x <= 32 ?  warp_max_value<float>(local_max): block_max_value<float>(local_max);
        if(threadIdx.x==0)
        {
            s_max=MAX(s_max,max_val);
        }
        __syncthreads();
        float local_sum = 0;
        for (int64_t index = start_idx; index < end_idx; index += blockDim.x) {
            float val = input[index];
            val *= scale;
            val = __expf(val - s_max);
            local_sum += val;
            output[index] = val;
        }

        float sum = blockDim.x <= 32 ? warp_max_value<float>(local_sum) : block_max_value<float>(local_sum);
        if (threadIdx.x == 0) {
            s_sum = sum;
        }
        __syncthreads();

        float to_mult = __fdividef((float)1.0, s_sum+1e-6f);
        for (int64_t index = index_start; index < index_end; index += blockDim.x) {
            float val = output[index];
            val *= to_mult;
            output[index] = (T)val;
        }
    }

    template<typename T>
    void scale_softmax(
        const T *input,
        T* output,
        const float scale,
        int64_t num_head,
        int64_t input_length
    ){
        uint32_t block_dim = std::min(seq_len, 256l);
        scale_softmax_kernel<<<num_head, block_dim>>>(output, input, scale, input_length);
    }




    /*
    scaleMaskSoftmaxKernel &
    scaleMaskSoftmax

    This kernel applies scaling (*1/sqrt(dk)), masking, and softmax to the input matrix (attention matrix).

	Input:
		- input: [num_heads, input_len, input_len]
		- scale: 1/sqrt(dk)
	Output:
		- output: [num_heads, input_len, input_len]
		  output[head][row] = softmax(masking(input[head][row] * scale))
*/
    template<typename T>
    __global__ void scale_softmax_mask_kernel(
        const T*input,
        T* output,
        float scale,
        uint64_t num_head,
        uint64_t input_length
    ){
        const uint64_t z=blockIdx.x;
        for(uint64_t y=0;y<input_length;y++)
        {
            __shared__ T s_max=0,s_sum=0;
            T local_max = -1e20f, local_sum = 0.0;
            for (uint64_t  x=threadIdx.x;x<input_length;x+=blockDim.x)
            {
                T value=input(INDEX_3D(num_head,z,input_length,y,input_length,x));
                value *= scale;
                value += r >= c ? 0 : -10000.0;
                output[INDEX_3D(num_head, z,input_length,y,input_length,x)] = val;
                local_max = MAX(local_max,value);
            }
            float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
		    if (threadIdx.x == 0) {
			s_max = max_val;
		    }
            __syncthreads();
            for (int64_t c = threadIdx.x; c < input_length; c += blockDim.x) {
                float val = output[INDEX_3D(num_head,z,input_length,y,input_length,x)];
                val = __expf(val - s_max);
                output[INDEX_3D(num_head,z,input_length,y,input_length,x)] = val;
                local_sum += val;
            }
            float sum = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
            if (threadIdx.x == 0) {
                s_sum = sum;
            }
            __syncthreads();
            float to_mult = __fdividef((float)1.0, s_sum+(float)(1e-6));
            for (int64_t c = threadIdx.x; c < input_length; c += blockDim.x) {
                float val = output[INDEX_3D(num_head,z,input_length,y,input_length,x)];
                val *= to_mult;
                output[INDEX_3D(num_head,z,input_length,y,input_length,x)] = val;
		}
        }
    }
   
    template<typename T>
    void scale_mask_softmax(
	T* output,
	const T* input,
	const float scale,
	const int64_t num_heads,
	const int64_t input_len
) {
	uint32_t block_dim = std::min(input_len, 256l);
    scale_softmax_mask_kernel<<<num_heads, block_dim>>>(
		output,
		input,
		scale,
		num_heads,
		input_len
	)
}
}
