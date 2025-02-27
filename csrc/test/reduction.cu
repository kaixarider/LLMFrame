#include <iostream>
#include <cuda_runtime.h>
#include "../util/reduction.cuh"  // 包含你的头文件

// 测试内核函数
__global__ void test_kernel(int* input, int* output)
{
    int tid = threadIdx.x;
    int val = input[tid];
    output[tid] = block_max_value<int>(val);  // 调用warp_max_value函数
}

// 主机代码
int main() {
    const int num_elements = 498;  // 设置测试的元素数量
    int h_input[num_elements], h_output[num_elements];  // 主机输入和输出数组
    int* d_input;
    int* d_output;

    // 初始化主机数据
    for (int i = 0; i < num_elements; i++) {
        h_input[i] = i;  // 随机填充输入数据
    }

    // 分配设备内存
    cudaMalloc((void**)&d_input, num_elements * sizeof(int));
    cudaMalloc((void**)&d_output, num_elements * sizeof(int));

    // 将输入数据从主机复制到设备
    cudaMemcpy(d_input, h_input, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    // 启动内核
    test_kernel<<<1, num_elements>>>(d_input, d_output);

    // 检查是否有错误
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    // 显示前10个结果
    for (int i = 0; i < 5; i++) {
        std::cout << "Input: " << h_input[i] << " - Output: " << h_output[i] << std::endl;
    }

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}