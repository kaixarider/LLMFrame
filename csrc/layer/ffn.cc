#include "ffn.h"
namespace layer{
    /*
    ffn:matrix multiple.Feed forward network with tensor parallelism
     
    */
   template<typename T>
   void ffn{
    T* output,
	T* input,

	T* fc1_weight,
	T* fc1_bias,
	T* fc2_weight,
	T* fc2_bias,

	int64_t batch_size,
	int64_t input_dim,
	int64_t inter_dim,
	int64_t output_dim,
	ActivationType activation_type,

	T* inter_buf,
    }{
        
    }
}