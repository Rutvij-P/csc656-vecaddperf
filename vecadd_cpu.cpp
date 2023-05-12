#include <iostream>
#include <chrono>

void add(int n, float* x, float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 26; // As instructed in the Note#1
    size_t bytes = N * sizeof(float);

    // Allocate memory
    float* x = new float[N];
    float* y = new float[N];

    // Initialize array
    for (int i = 0; i < N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Time the kernel
    auto start = std::chrono::high_resolution_clock::now();

    // Perform Vector addition
    add(N, x, y);

    // Time the kernel
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end - start;
    std::cout << "Vector addition took " << duration_ms.count() << " ms" << std::endl;

    // Free memory
    delete[] x;
    delete[] y;

    return 0;
}