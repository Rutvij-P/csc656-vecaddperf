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
    int N = 1 << 26; // 6M elements
    float* x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Prefetch memory to GPU
    int deviceID = 0;
    cudaMemPrefetchAsync((void*)x, N * sizeof(float), deviceID);
    cudaMemPrefetchAsync((void*)y, N * sizeof(float), deviceID);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocksPerGrid, threadsPerBlock>>>(N, x, y);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    // Cleanup
    cudaFree(x);
    cudaFree(y);

    return 0;
}
