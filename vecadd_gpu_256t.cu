#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void add(int n, float* x, float* y)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
    {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 26; // 64M elements
    size_t bytes = N * sizeof(float);

    // Allocate memory on the device
    float* d_x, *d_y;
    cudaMallocManaged(&d_x, bytes);
    cudaMallocManaged(&d_y, bytes);

    // Initialize input vectors
    for (int i = 0; i < N; ++i)
    {
        d_x[i] = 1.0f;
        d_y[i] = 2.0f;
    }

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, d_x, d_y);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
