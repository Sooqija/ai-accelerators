#include <iostream>
#include <vector>
#include <string>
#include <random>

#include <omp.h>
#include <immintrin.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cassert>

#define BLOCK_SIZE 32


// initialization functions
std::vector<float> initMatrix(int N, float min = 0.0f, float max = 1.0f) {
    std::vector<float> matrix(N * N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (int i = 0; i < N * N; ++i) {
        matrix[i] = dist(gen);
    }

    return matrix;
}

// helper functions
void printMatrix(std::vector<float>& matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", matrix[j + i * N]);
        }
        printf("\n");
    }
}

__device__ void printMatrix(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", matrix[j + i * N]);
        }
        printf("\n");
    }
}

float calcAbsDiff(std::vector<float>& A, std::vector<float>& B, const int N) {

    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            max_diff = std::max(abs(A[i * N + j] - B[i * N + j]), max_diff);
        }
    }
    return max_diff;
}

float batchedAbsDiff(std::vector<std::vector<float>> batchedC0, std::vector<std::vector<float>> batchedC1, const int m, const int output_size) {
    float batched_max_diff = 0.0f;
    for (int i = 0; i < m; i++) {
        assert(batchedC0[i].size() == output_size * output_size);
        assert(batchedC0[i].size() == batchedC1[i].size());
        float diff = calcAbsDiff(batchedC0[i], batchedC1[i], output_size);
        batched_max_diff = std::max(batched_max_diff, diff);
    }
    return batched_max_diff;
}

// Conv2D functions
std::vector<float> sequentialConv2D(const std::vector<float>& A, const int N, const std::vector<float>& K, const int k) {
    const int output_size = N - k + 1;
    std::vector<float> C(output_size * output_size);
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            float sum = 0.0f;
            // Обращаемся к K последовательно, что может помочь с кэшированием
            for (int m = 0; m < k * k; m++) {
                int ii = m / k;
                int jj = m % k;
                sum += A[(i + ii) * N + (j + jj)] * K[m];
            }
            C[i * output_size + j] = sum;
        }
    }
    return C;
}

std::vector<std::vector<float>> batchedSequentialConv2D(const std::vector<std::vector<float>>& batchedA, const int N, std::vector<std::vector<float>>& batchedK, const int k, const int m) {
    const int output_size = N - k + 1;
    std::vector<std::vector<float>> batchedC(m, std::vector<float>(output_size * output_size));
    for (int i = 0; i < m; i++) {
        batchedC[i] = sequentialConv2D(batchedA[i], N, batchedK[i], k);
    }
    return batchedC;
}

std::vector<std::vector<float>> batchedParalelConv2d(const std::vector<std::vector<float>>& batchedA, const int N, std::vector<std::vector<float>>& batchedK, const int k, const int m) {
    const int output_size = N - k + 1;
    std::vector<std::vector<float>> batchedC(m, std::vector<float>(output_size * output_size));
# pragma omp parallel for
    for (int i = 0; i < m; i++) {
        batchedC[i] = sequentialConv2D(batchedA[i], N, batchedK[i], k);
    }
    return batchedC;
}

__global__ void Conv2D(const float* __restrict__ A, const float* __restrict__ K, float* C, const int N, const int k) {
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    const int output_size = N - k + 1;

    // shared kernel
    extern __shared__ float shared_K[];
    if (threadIdx.y * blockDim.x + threadIdx.x < k * k) {
        shared_K[threadIdx.y * blockDim.x + threadIdx.x] = 
            K[threadIdx.y * blockDim.x + threadIdx.x];
    }
    __syncthreads();

    if (row < output_size && col < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                sum += A[(row + i) * N + (col + j)] * shared_K[i * k + j];
            }
        }
        C[row * output_size + col] = sum;
    }
}

std::vector<std::vector<float>> simtConv2D(const std::vector<std::vector<float>>& batchedA, const int N, std::vector<std::vector<float>>& batchedK, const int k, const int m, const int output_size, float* time) {
    std::vector<std::vector<float>> batchedC(m, std::vector<float>(output_size * output_size));
    std::vector<float> C(output_size * output_size);
    cudaEvent_t start_time, end_time;
    cudaError_t cudaStatus;

    float* d_A;
    float* d_K;
    float* d_C;

    cudaMalloc(&d_A, m * N * N * sizeof(float));
    cudaMalloc(&d_K, m * k * k * sizeof(float));
    cudaMalloc(&d_C, m * output_size * output_size * sizeof(float));

    int threads_per_block = BLOCK_SIZE;
    dim3 blockDim(threads_per_block, threads_per_block);
    dim3 gridDim((output_size + blockDim.x - 1) / blockDim.x, (output_size + blockDim.y - 1) / blockDim.y);
    size_t shared_mem_size = k * k * sizeof(float);
    if (k * k > blockDim.x * blockDim.y) { throw std::runtime_error("Kernel size is too large for the block size"); }
    
    cudaEventCreate(&start_time); cudaEventCreate(&end_time);
    cudaEventRecord(start_time);

    for (int i = 0; i < m; i++) {
        cudaMemcpy(d_A, batchedA[i].data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, batchedK[i].data(), k * k * sizeof(float), cudaMemcpyHostToDevice);

        Conv2D<<<gridDim, blockDim, shared_mem_size>>>(d_A, d_K, d_C, N, k);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { printf(cudaGetErrorString(cudaStatus)); }
        cudaDeviceSynchronize();
        cudaMemcpy(C.data(), d_C, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);
        batchedC[i] = C;
    }

    cudaDeviceSynchronize();
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    cudaEventElapsedTime(time, start_time, end_time);

    cudaFree(d_A);
    cudaFree(d_K);
    cudaFree(d_C);

    return batchedC;
}

std::vector<std::vector<float>> streamSimtConv2D(const std::vector<std::vector<float>>& batchedA, const int N, std::vector<std::vector<float>>& batchedK, const int k, const int m, const int output_size, float* time) {
    std::vector<std::vector<float>> batchedC(m, std::vector<float>(output_size * output_size));
    cudaEvent_t start_time, end_time;
    cudaError_t cudaStatus;

    float* d_A;
    float* d_K;
    float* d_C;

    cudaMalloc(&d_A, m * N * N * sizeof(float));
    cudaMalloc(&d_K, m * k * k * sizeof(float));
    cudaMalloc(&d_C, m * output_size * output_size * sizeof(float));

    // cuda streams
    const int num_streams = 8;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads_per_block = BLOCK_SIZE;
    dim3 blockDim(threads_per_block, threads_per_block);
    dim3 gridDim((output_size + blockDim.x - 1) / blockDim.x, (output_size + blockDim.y - 1) / blockDim.y);
    size_t shared_mem_size = k * k * sizeof(float);
    if (k * k > blockDim.x * blockDim.y) { throw std::runtime_error("Kernel size is too large for the block size"); }
    
    cudaEventCreate(&start_time); cudaEventCreate(&end_time);
    cudaEventRecord(start_time);

    for (int i = 0; i < m; i++) {
        int stream_idx = i % num_streams;
        cudaStreamWaitEvent(streams[stream_idx], start_time, 0);

        cudaMemcpyAsync(d_A, batchedA[i].data(), N * N * sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]);
        cudaMemcpyAsync(d_K, batchedK[i].data(), k * k * sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]);

        Conv2D<<<gridDim, blockDim, shared_mem_size, streams[stream_idx]>>>(d_A, d_K, d_C, N, k);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) { printf(cudaGetErrorString(cudaStatus)); }

        cudaMemcpyAsync(batchedC[i].data(), d_C, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost, streams[stream_idx]);
    }

    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    cudaEventElapsedTime(time, start_time, end_time);

    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_A);
    cudaFree(d_K);
    cudaFree(d_C);

    return batchedC;
}

int main(int argc, char** argv) {
    int n = 1024; // input size
    int k = 3; // kernel size
    int m = 16; // batch size
    double t0, t1;
    if (argc > 1) {
        n = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        m = std::stoi(argv[3]);
    }
    int output_size = n - k + 1;
    float batched_max_diff;

    // Initialization
    std::vector<std::vector<float>> batchedA(m, std::vector<float>(n * n));
    std::vector<std::vector<float>> batchedK(m, std::vector<float>(k * k));
    for (int i = 0; i < m; i++) {
        batchedA[i] = initMatrix(n);
        batchedK[i] = initMatrix(k);
    }

    // Sequential
    t0 = omp_get_wtime();
    std::vector<std::vector<float>> batchedC0 = batchedSequentialConv2D(batchedA, n, batchedK, k, m);
    t1 = omp_get_wtime();
    printf("Sequential: %f\n", t1 - t0);

    //Parallel
    t0 = omp_get_wtime();
    std::vector<std::vector<float>> batchedC1 = batchedParalelConv2d(batchedA, n, batchedK, k, m);
    t1 = omp_get_wtime();
    batched_max_diff = batchedAbsDiff(batchedC0, batchedC1, m, output_size);
    printf("Parallel: %f sec (diff: %f)\n", t1 - t0, batched_max_diff);

    // SIMT
    float cuda_time;
    std::vector<std::vector<float>> batchedC2 = simtConv2D(batchedA, n, batchedK, k, m, output_size, &cuda_time);

    batched_max_diff = batchedAbsDiff(batchedC0, batchedC2, m, output_size);
    printf("SIMT: %f sec (diff: %f)\n", cuda_time / 1'000.0, batched_max_diff);

    // Streams
    cuda_time = 0.0;
    std::vector<std::vector<float>> batchedC3 = streamSimtConv2D(batchedA, n, batchedK, k, m, output_size, &cuda_time);

    batched_max_diff = batchedAbsDiff(batchedC0, batchedC3, m, output_size);
    printf("Streams: %f sec (diff: %f)\n", cuda_time / 1'000.0, batched_max_diff);

    return 0;
}
