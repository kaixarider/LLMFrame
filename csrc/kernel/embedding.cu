//TODO:把embedding搞明白
#include "embedding.h"
namespace kernel{
    template<typename T, bool IS_NEOX>
    __inline__ __device__ void apply_token_rotary_embedding
    (
        T* __restrict__ array,const T* __restrict__ cos_ptr,
        const T* __restrict__ sin_ptr,int rotary_offset,
        int embedding_dim
    )
    {
        int x_index,y_index;
        T cos,sin;
        if(IS_NEOX)
        {
            x_index=rotary_offset;
            y_index=rotary_offset+embedding_dim;
            cos=__ldg(cos_ptr+x_index);
            sin=__ldg(sin_ptr+x_index);
        }
        else{
            x_index=rotary_offset*2;
            y_index=rotary_offset*2+1;
            cos=__ldg(cos_ptr+x_index/2);
            sin=__ldg(sin_ptr+y_index/2);
        }
        const T x=array[x_index];
        const T y=array[y_index];
        array[x_index]=x*cos-y*sin;
        array[y_index]=y*cos-x*sin;
    }
    template<typename T,bool IS_NEOX>
    __inline__ __device__ void apply_rotary_embedding(
        T* __restrict__ q,//[batchsize,seq_len,num_head,
                          //head_size] or [num_token,
                          //num_head,head_size]
        T* __restrict__ k,//[batchsize,seq_len,num_head,
        //head_size] or [num_token,
        //num_head,head_size]
        const T* cache_ptr,const int num_heads,const int head_size,
        const int num_kv_heads, const int rot_dim, const int token_idx,
        const int64_t query_stride, const int64_t key_stride
    )
    {
        const int embed_dim=rot_dim/2;
        const T* cos_ptr=cache_ptr;
        const T* sin_ptr=cache_ptr+embed_dim;
        const int nq = num_heads * embed_dim;
        for (int i = threadIdx.x; i < nq; i += blockDim.x) {
            const int head_idx = i / embed_dim;
            const int64_t token_head = token_idx * query_stride + head_idx * head_size;
            const int rot_offset = i % embed_dim;
            apply_token_rotary_embedding<scalar_t, IS_NEOX>(
                q+ token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
        }

        const int nk = num_kv_heads * embed_dim;
        for (int i = threadIdx.x; i < nk; i += blockDim.x) {
            const int head_idx = i / embed_dim;
            const int64_t token_head = token_idx * key_stride + head_idx * head_size;
            const int rot_offset = i % embed_dim;
            apply_token_rotary_embedding<scalar_t, IS_NEOX>(
                k + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }
    }
    template <typename scalar_t, bool IS_NEOX>
    __global__ void rotary_embedding_kernel(
        const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                                // [num_tokens]
        scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                       // head_size] or [num_tokens, num_heads,
                                       // head_size]
        scalar_t* __restrict__ key,  // [batch_size, seq_len, num_kv_heads,
                                     // head_size] or [num_tokens, num_kv_heads,
                                     // head_size]
        const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                     // 2]
        const int rot_dim, const int64_t query_stride, const int64_t key_stride,
        const int num_heads, const int num_kv_heads, const int head_size) {
      // Each thread block is responsible for one token.
      const int token_idx = blockIdx.x;
      int64_t pos = positions[token_idx];
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
    
      apply_rotary_embedding<scalar_t, IS_NEOX>(
          query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
          token_idx, query_stride, key_stride);
    }
    template <typename scalar_t, bool IS_NEOX>
__global__ void batched_rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [batch_size, seq_len] or
                                            // [num_tokens]
    scalar_t* __restrict__ query,           // [batch_size, seq_len, num_heads,
                                   // head_size] or [num_tokens, num_heads,
                                   // head_size]
    scalar_t* __restrict__ key,  // [batch_size, seq_len, num_kv_heads,
                                 // head_size] or [num_tokens, num_kv_heads,
                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int64_t* __restrict__ cos_sin_cache_offsets,  // [batch_size, seq_len]
                                                        // or [num_tokens]
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int num_heads, const int num_kv_heads, const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  int64_t cos_sin_cache_offset = cos_sin_cache_offsets[token_idx];
  const scalar_t* cache_ptr =
      cos_sin_cache + (cos_sin_cache_offset + pos) * rot_dim;

  apply_rotary_embedding<scalar_t, IS_NEOX>(
      query, key, cache_ptr, head_size, num_heads, num_kv_heads, rot_dim,
      token_idx, query_stride, key_stride);
}
}