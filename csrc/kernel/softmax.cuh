#pragma once
#include<stdint.h>
namespace kernel{
    template<typename T>
    /*softmax(Array/scale),scale is √dk 
        -input:[seq_len,seq_len]
        -output:[seq_len,seq_len]
    */
    void scale_softmax(
        const T *input,
        T* output,
        const float scale,
        int64_t num_head,
        int64_t input_length
    );

    
       /*softmax(Array/scale),scale is √dk 
        -input:[seq_len,seq_len]
        -output:[seq_len,seq_len]
    */ 
   template<typename T>
   void scale_mask_softmax(
   T* output,
   const T* input,
   const float scale,
   const int64_t num_heads,
   const int64_t input_len
);
template void kernel::scale_softmax<float>(const float*, float*, float, int64_t, int64_t);
}
