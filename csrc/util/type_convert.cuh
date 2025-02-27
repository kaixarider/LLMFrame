#pragma once

#include<torch/all.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
template <typename torch_type>
struct _typeConvert {
  static constexpr bool exists = false;
};