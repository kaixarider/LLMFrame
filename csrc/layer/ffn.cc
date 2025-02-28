#include "ffn.h"
#include "../util/cublas_wrapper.h"
namespace layer{
    /*
    ffn:matrix multiple.Feed forward network with tensor parallelism
     Artchitecture:
	 | Linear 1
	 | inter = input 1 * weight 1 T+bias 1
	 | -input 1:[token num,hidden_size]
	 | -weight 1:[inter_dim/tp_size,hidden_size]
	 | -bias 1:[inter_dim/tp_size]
	 | -inter:[token num,inter_dim/tp_size]
     ⬇ 
	 | Activation
	 |
	 ⬇Linear 2
	 | output=inter * weight 2 
     | -inter[token_num,inter_dim/tp_size]
	 | -weight[output_dim,inter_dim/tp_size]
	 | -output [token num,hidden size]
	 |
	 | Reduce
	 |
	 ⬇add_bias
	  */
   template<typename T>
   void ffn{
    T* __restrict__ output,
	T* __restrict__ input,

	T* __restrict__ fc1_weight,
	T* __restrict__ fc1_bias,
	T* __restrict__fc2_weight,
	T* __restrict__ fc2_bias,

	int64_t token_num,
	int64_t input_dim,
	int64_t inter_dim,
	int64_t output_dim,
	ActivationType activation_type,

	T* inter_buf,	

	util::CublasWrapper cublas_wrapper,
	util::NcclComm nccl_comm
    }{
		//Linear 1
        cublas_wrapper.gemm(
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			token_num,
			inter_dim/nccl_comm.size,
			input_dim,
			input,
			fc1_weight,
			inter_buf
		);
		sync_check_cuda_error();
		//activation
		kernel::fused_addbias_batched_activation(output,output,fc1_bias,token_num,inter_dim/nccl_comm.size,activation_type);
		//Linear 2
		cublas_wrapper.gemm(
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			token_num,
			output_dim,
			inter_dim/nccl_comm.size,
			inter_buf,
			fc2_weight,
			output
		)
		sync_check_cuda_error();
		if(nccl_comm.size!=1)
		{
			util::stNcclAllReduce(
				output,
				output,
				batch_size * output_dim,
				util::stNcclGetDataType<T>(),
				ncclSum,
				nccl_comm.comm,
				nccl_comm.stream
			);
		}
		sync_check_cuda_error();

		// Addbias
		kernel::addbiasBatched(output, output, fc2_bias, batch_size, output_dim);
		sync_check_cuda_error();
    }
	template void ffn(
		float* output, float* input,
		float* fc1_weight, float* fc1_bias,
		float* fc2_weight, float* fc2_bias,
		int64_t batch_size, int64_t input_dim, int64_t inter_dim, int64_t output_dim,
		ActivationType activation_type,
		float* inter_buf, util::CublasWrapper cublas_wrapper,
		util::NcclComm nccl_comm
	);
	
	template void ffn(
		half* output, half* input,
		half* fc1_weight, half* fc1_bias,
		half* fc2_weight, half* fc2_bias,
		int64_t batch_size, int64_t input_dim, int64_t inter_dim, int64_t output_dim,
		ActivationType activation_type,
		half* inter_buf, util::CublasWrapper cublas_wrapper,
		util::NcclComm nccl_comm
	);
	
}