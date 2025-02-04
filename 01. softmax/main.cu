#include <iostream>
#include <vector>
#include <string>
#include <random>

#include <omp.h>
#include <immintrin.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_SIZE 256


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

std::vector<float> zeros(int N) {
    std::vector<float> matrix;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix.push_back(0);
        }
    }
    return matrix;
}

// helper functions
void printMatrix(std::vector<float>& matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

float calcAbsDiff(std::vector<float>& A, std::vector<float>& B, const int N) {
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (abs(A[j + i * N] - B[j + i * N]) > max_diff)
                max_diff = abs(A[i * N + j] - B[i * N + j]);
        }
    }
    return max_diff;
}

// SoftMax functions

void sequenctialSoftMax(const std::vector<float>& A, std::vector<float>& B, const int N) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;

        for (int j = i*N; j < i*N + N; j++) {
            B[j] = expf(A[j]);
            sum += B[j];
        }
        const float inv_sum = 1.0f / sum;
        for (int j = i*N; j < i*N + N; j++) {
            B[j] *= inv_sum;
        }
    }
}

void parallelSoftMax(std::vector<float>& A, std::vector<float>& B, const int N) {
    # pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;

        for (int j = i*N; j < i*N + N; j++) {
            B[j] = expf(A[j]);
            sum += B[j];
        }
        const float inv_sum = 1.0f / sum; 
        for (int j = i*N; j < i*N + N; j++) {
            B[j] *= inv_sum;
        }
    }
}

void simdSoftMax(const std::vector <float>& __restrict__ A, std::vector <float>& __restrict__ B, const int N) {
    const int VECTOR_SIZE = sizeof(__m512) / sizeof(float);
    const int vectorized_part = N / VECTOR_SIZE * VECTOR_SIZE;
    
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        __m512 vector_sum = _mm512_setzero_ps();
        for (int j = 0; j < vectorized_part; j+=VECTOR_SIZE) {
            _mm512_storeu_ps(&B[i * N + j], _mm512_exp_ps(_mm512_loadu_ps(&A[i * N + j])));
            vector_sum = _mm512_add_ps(vector_sum, _mm512_loadu_ps(&B[i * N + j]));
        }

        float vector_sum_array [VECTOR_SIZE];
        _mm512_storeu_ps(vector_sum_array, vector_sum);
        for (int j = 0; j < VECTOR_SIZE; j++) {
            sum += vector_sum_array[j];
        }
        
        for (int j = vectorized_part; j < N; j++) {
            B[i * N + j] = expf(A[i * N + j]);
            sum += B[i * N + j];
        }

        __m512 inv_sum = _mm512_set1_ps(1.0f / sum);
        for (int j = 0; j < vectorized_part; j+=VECTOR_SIZE) {
            _mm512_storeu_ps(&B[i * N + j], _mm512_mul_ps(_mm512_loadu_ps(&B[i * N + j]), inv_sum));
        }
        
        for (int j = vectorized_part; j < N; j++) {
            B[i * N + j] *= _mm512_cvtss_f32(inv_sum);
        }
    }
}

void parallelSimdSoftMax(std::vector<float>& __restrict__ A, std::vector<float>& __restrict__ B, const int N) {
    const int VECTOR_SIZE = sizeof(__m512) / sizeof(float);
    const int vectorized_part = N / VECTOR_SIZE * VECTOR_SIZE;

    # pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        __m512 vector_sum = _mm512_setzero_ps();
        for (int j = 0; j < vectorized_part; j+=VECTOR_SIZE) {
            _mm512_storeu_ps(&B[i * N + j], _mm512_exp_ps(_mm512_loadu_ps(&A[i * N + j])));
            vector_sum = _mm512_add_ps(vector_sum, _mm512_loadu_ps(&B[i * N + j]));
        }

        float vector_sum_array [VECTOR_SIZE];
        _mm512_storeu_ps(vector_sum_array, vector_sum);
        for (int j = 0; j < VECTOR_SIZE; j++) {
            sum += vector_sum_array[j];
        }
        
        for (int j = vectorized_part; j < N; j++) {
            B[i * N + j] = expf(A[i * N + j]);
            sum += B[i * N + j];
        }

        __m512 inv_sum = _mm512_set1_ps(1.0f / sum);
        for (int j = 0; j < vectorized_part; j+=VECTOR_SIZE) {
            _mm512_storeu_ps(&B[i * N + j], _mm512_mul_ps(_mm512_loadu_ps(&B[i * N + j]), inv_sum));
        }
        
        for (int j = vectorized_part; j < N; j++) {
            B[i * N + j] *= _mm512_cvtss_f32(inv_sum);
        }
    }
}

__global__ void SoftMax(const float* __restrict__ A, float* B, const int N) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int thread_num = blockDim.x;
    const int subblock_size = ceilf(float(N) / float(thread_num));

    extern __shared__ float shared_sum[];
    for (int j = tid*subblock_size; j < (tid+1)*subblock_size && j < N; j++) {
        B[i * N + j] = __expf(A[i * N + j]);
        shared_sum[tid] += B[i * N + j];
    }
    __syncthreads();
    
    __shared__ float sum;
    atomicAdd(&sum, shared_sum[tid]);

    for (int j = tid*subblock_size; j < (tid+1)*subblock_size && j < N; j++) {
        B[i * N + j] = __fdividef(B[i * N + j], sum);
    }
}

float simtSoftMax(std::vector<float>& A, std::vector<float>& B, const int N) {
    cudaEvent_t start_time, end_time; float time;
    
    float* d_A;
    float* d_B;

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = BLOCK_SIZE;
    int num_blocks = N;

    cudaEventCreate(&start_time); cudaEventCreate(&end_time);
    cudaEventRecord(start_time);
    SoftMax<<<num_blocks, threads_per_block>>>(d_A, d_B, N);
    cudaDeviceSynchronize();
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    cudaEventElapsedTime(&time, start_time, end_time);

    cudaMemcpy(B.data(), d_B, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    return time / 1'000.0;
}

int main(int argc, char** argv) {
    int n = 1024;
    if (argc > 1) {
        n = std::stoi(argv[1]);
    }
    std::vector <float> A = initMatrix(n);
    std::vector <float> B = zeros(n);
    std::vector <float> seqB = zeros(n);
    double start_time, end_time; float diff;

    // Sequential
    start_time = omp_get_wtime();
    sequenctialSoftMax(A, seqB, n);
    end_time = omp_get_wtime();
    printf("Sequential: %f sec\n", end_time - start_time);

    // Parallel
    B = zeros(n);
    start_time = omp_get_wtime();
    parallelSoftMax(A, B, n);
    end_time = omp_get_wtime();
    diff = calcAbsDiff(seqB, B, n);
    printf("Parallel: %f sec (diff: %f)\n", end_time - start_time, diff);

    // SIMD
    B = zeros(n);
    start_time = omp_get_wtime();
    simdSoftMax(A, B, n);
    end_time = omp_get_wtime();
    diff = calcAbsDiff(seqB, B, n);
    printf("SIMD: %f sec (diff: %f)\n", end_time - start_time, diff);

    // Parallel+SIMD
    B = zeros(n);
    start_time = omp_get_wtime();
    parallelSimdSoftMax(A, B, n);
    end_time = omp_get_wtime();
    diff = calcAbsDiff(seqB, B, n);
    printf("Parallel+SIMD: %f sec (diff: %f)\n", end_time - start_time, diff);

    // CUDA/SIMT
    B = zeros(n);
    double time = simtSoftMax(A, B, n);
    diff = calcAbsDiff(seqB, B, n);
    printf("SIMT: %f sec (diff: %f)\n", time, diff);

    return 0;
}
