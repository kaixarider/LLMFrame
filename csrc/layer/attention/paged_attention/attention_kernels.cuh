#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include "attention_generic.cuh"
#include "dtype_fp8.cuh"
#define WARP_SIZE 32
#define DIVIDE_ROUND_UP (a,b) (((a)+(b)-1)/(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))
namespace paged_attention{
template <int WARP_NUM>
__inline__ __device__ float block_sum(
    float sum,float*red_smem
)
{
    int warp=threadIdx.x/WARP_SIZE;
    int lane=threadIdx.x%WARP_SIZE;
    #pragma unroll
    for(int mask=16;mask>0;mask/=2)
    {
        sum+=__shfl_xor_sync(sum,mask);
    }
    if(lane==0)
    {
        red_smem[warp]=sum;
    }
    __syncthreads();

  // The warps compute the final sums.
  if (lane < WARP_NUM) {
    sum = red_smem[lane];
  }
  #pragma unroll
  for (int mask = WARP_NUM / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_xor_sync(sum, 0);
}
template<typename matrix_t,typename cache_t,int head_size,
int BLOCK_SIZE,int NUM_THREADS,paged_attention::Fp8KVCacheDataType KV_DTYPE,
bool IS_BLOCK_SPARSE,int PARTITION_SIZE=0>
__device__ void paged_attention_kernel(
  float* __restrict__ exp_sum,//[num_seqs,num_heads,max_num_partitions]
  float* __restrict__ max_logits,//[num_seqs,num_heads,max_num_partitions]
  matrix_t *__restrict__ out,//[batch_size,num_heads,partition,head_dim] 为什么没有seq_len?
  const matrix_t * __restrict__ q,//[batch_size,num_head,head_dim]
  const cache_t * __restrict__ k_cache,//[num_blocks, num_kv_heads,
                                      // head_size/x, block_size, x] x是什么？
  const cache_t * __restrict__ v_cache,//[num_blocks, num_kv_heads,head_size, block_size]
  const int num_kv_heads,
  const float scale,const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ seq_lens,      // [num_seqs]
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes,  // [num_heads]
  const int q_stride, const int kv_block_stride, const int kv_head_stride,
  const float* k_scale, const float* v_scale, const int tp_rank,
  const int blocksparse_local_blocks, const int blocksparse_vert_stride,
  const int blocksparse_block_size, const int blocksparse_head_sliding_step
){
  const int partition_idx=blockIdx.z;
  const int seq_idx=blockIdx.y;
  const int head_idx=blockIdx.x;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITION=PARTITION_SIZE>0;
  const int seq_len=seq_lens[seq_idx];
  if(USE_PARTITION&&partition_idx*PARTITION_SIZE>seq_len)
  {
    return;//no tokens to handle
  }
  //block range each partition to handle,[start,end)
  const int num_seq_blocks=DIVIDE_ROUND_UP(seq_len,BLOCK_SIZE);
  const int num_blocks_each_partition=USE_PARTITION?num_blocks/PARTITION_SIZE:num_seq_blocks;
  const int start_block_idx=USE_PARTITION?partition_idx*num_blocks_each_partition:0;
  const int end_block_idx=MIN(start_block_idx + num_blocks_each_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;
  //tokens range each partition to handle
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx=MIN(seq_len,start_token_idx+BLOCK_SIZE*num_blocks);

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS=NUM_THREADS/THREAD_GROUP_SIZE;
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);//num_threads is divisible by thread_group_size
  constexpr int NUM_TOKENS_PER_THREAD_GROUP=DIVIDE_ROUND_UP(BLOCK_SIZE,WARP_SIZE);//
  const int num_warps=NUM_THREADS/WARP_SIZE;
  const int thread_idx=threadIdx.x;
  const int warp_idx=threadIdx.x/WARP_SIZE;
  const int lane=threadIdx%WARP_SIZE;

  const int num_heads=gridDim.x;
  const int query_num_per_kv_head=num_heads/num_kv_heads;
  const int kv_head_idx=head_idx/query_num_per_kv_head;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
  //convert k and q array into a new type of vector
  //so that threads in a group can fetch and compute
  //16 bytes at a time
  const int vector_type_size=MAX(16/(THREAD_GROUP_SIZE*sizeof(matrix_t)),1);
  using K_vec = typename Vec<scalar_t, vector_type_size>::Type;
  using Q_vec = typename Vec<scalar_t, vector_type_size>::Type;
  using Quant_vec = typename Vec<cache_t, vector_type_size>::Type;
  
  constexpr int NUM_ELEMS_PER_THREAD=HEAD_SIZE/THREAD_GROUP_SIZE;//elems per thread group
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / vector_type_size;
  
  const int thread_group_idx=thread_idx/THREAD_GROUP_SIZE;
  const int thread_group_offset=thread_idx%THREAD_GROUP_SIZE;
  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const matrix_t* q_ptr=q+seq_idx*q_stride+head_idx*head_size;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];//
  #pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads(); 
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];
  
  

}
}
