#pragma once
#include "util/cuda_utils.h"
namespace kernel{
    template<typename T>
void embedAndPosEncodeBatched(
	T* result,
	const int64_t* token_ids,
	const int64_t* position_ids,
	const T* embed_tokens_weight,
	const T* embed_positions_weight,
	const int64_t num_tokens,
	const int64_t hidden_size
);
}