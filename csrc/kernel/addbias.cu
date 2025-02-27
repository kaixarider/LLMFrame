#include "addbias.h"
#include<util/calculate.h>

namespace kernel{
    /*
        -input: the input array, [size]
        -bias: the bias array, [size]
        -size: the size of the input and bias array
        -output: the output array, [size]
            output[i] = input[i] + bias[i]
    */
    template<typename T>
    __global__ void addbias_kernel(
        T *output,
        const T* input,
        const T* bias,
        const int64_t size
    )
    {
        typedef std::conditional_t<std::is_same<T,half>::value,half2,float2> T2;
        #pragma unroll 
        for(int64_t i = blockIdx.x*blockDim.x+threadIdx.x;i<size/2;i+=blockDim.x*gridDim.x)
        {
            T2 input_elem = ((const T2*)input)[i];
            T2 bias_elem = ((const T2*)bias)[i];
            T2 result_elem = {input_elem.x+bias_elem.x,input_elem.y+bias_elem.y};
            ((T2*)output)[i] = result_elem;
        }
    }
    template<typename T>
    void addbias(
        T*output,
        const T*input,
        const T*bias,
        const int64_t size
    )
    {
        const uint32_t blocksize =256;
        const uint32_t gridsize=MIN(DIVIDE_ROUND_UP(size/2,blocksize),16384);
        addbias_kernel<T><<<gridsize,blocksize>>>(output,input,bias,size);
    }
    /*
        -input: the input array, [batch_size, size]
        -bias: the bias array, [size]
        -batch_size: the batch size
        -size:the size of the bias array
        -output: the output array, [batch_size, size]
            output[i][j] = input[i][j] + bias[j]
    */
   template<typename T>
   __global__ void addbias_batch_kernel(
    T* output,//[batchsize,size]
    const T* bias,//[size]
    const T* input,//[batchsize,size],input[i][j] is i*size+j
    const uint64_t size,
    const uint64_t batchsize
   )
   {
    typedef std::conditional_t<std::is_same<T,half>::value,half2,float2> T2;
    #pragma unroll
    for(uint64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<batchsize*size/2;i+=blockDim.x*gridDim.x)
    {
        const uint64_t s =i%(size/2);
        T2 input_elem = ((const T2*)input)[i];
        T2 bias_elem = ((const T2*)bias)[s];
        ((T2*)output)[i]=input_elem+bias_elem;
    }
   }
   
   template<typename T>
   void addbias_batch(
    T* output,//[batchsize,size]
    const T* bias,//[size]
    const T* input,//[batchsize,size],input[i][j] is i*size+j
    const uint64_t size,
    const uint64_t batchsize
   )
   {
    const uint64_t blocksize=256;
    const uint64_t gridsize=MIN(DIVIDE_ROUND_UP(size/2,blocksize),16384);
    addbias_batch_kernel<T><<<blocksize,gridsize>>>(output,bias,input,size,batchsize);
   }
}