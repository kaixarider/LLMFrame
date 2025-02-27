#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cassert>

// 引入你定义的头文件和核函数
#include "../kernel/softmax.cuh"

// 检查 CUDA 调用的错误
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    }

template <typename T>
void verify_results(const T* output, const T* expected, int64_t length) {
    for (int i = 0; i < length; i++) {
        assert(fabs(output[i] - expected[i]) < 1e-5);
    }
}

int main() {
    // 假设参数
    const int64_t num_head = 2;
    const int64_t input_length = 4;
    const float scale = 1.0f;

    // 输入数据
    float h_input[8] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};  // 2个head，每个head 4个长度
    float h_output[8] = {0.0};  // 存储输出结果

    // 计算预期结果，softmax 输出是归一化的，可以通过手动计算得到
    float h_expected[8] = {0.035, 0.096, 0.261, 0.608, 0.035, 0.096, 0.261, 0.608};

    // 分配设备内存
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(h_input)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(h_output)));

    // 将输入数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input, sizeof(h_input), cudaMemcpyHostToDevice));

    // 调用scale_softmax进行测试
    kernel::scale_softmax(d_input, d_output, scale, num_head, input_length);

    // 将结果从设备复制回主机
    CUDA_CHECK(cudaMemcpy(h_output, d_output, sizeof(h_output), cudaMemcpyDeviceToHost));

    // 验证结果
    verify_results(h_output, h_expected, 8);

    std::cout << "Softmax test passed!" << std::endl;

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
