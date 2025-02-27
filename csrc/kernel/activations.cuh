#pragma once

#include <cassert>
#include<math.h>
namespace kernel{
    enum class ActivationType {
        RELU,
        SILU,
        GELU
    };
    template <typename T, ActivationType activation_type>
    __forceinline__ __device__ T ApplyActivation(const T &x){
        if (activation_type==ActivationType::RELU){
            return x>(T)0?x:(T)0;
        }
        else if (activation_type==ActivationType::SILU){
            return (T)((float)x / (1.0f + __expf((float)-x)));
        }
        else if (activation_type==ActivationType::GELU)
        {
            const float f = (float)x;
            constexpr float ALPHA = M_SQRT1_2;
            return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
        }
        else{
            assert(false);
        }
    }
}