#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>

// Vector addition using OpenMP GPU offloading
// This function offloads computation to GPU device 0
// Uses explicit data mapping for efficient memory transfer
void vector_add(const std::vector<float>& a, 
                const std::vector<float>& b,
                std::vector<float>& result,
                size_t size) {
    
    // Get raw pointers for OpenMP mapping
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    float* result_ptr = result.data();
    
    // Offload computation to GPU device 0
    // map(to:) - transfer input data to device
    // map(from:) - transfer result back from device
    // teams distribute - create teams of threads for GPU execution
    // parallel for - parallelize loop iterations across threads
    #pragma omp target teams distribute parallel for \
        device(0) \
        map(to: a_ptr[0:size], b_ptr[0:size]) \
        map(from: result_ptr[0:size])
    for (size_t i = 0; i < size; i++) {
        result_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

// Dot product using OpenMP GPU offloading
// Demonstrates reduction operation on GPU
// Returns scalar result computed on device
float dot_product(const std::vector<float>& a,
                  const std::vector<float>& b,
                  size_t size) {
    
    float result = 0.0f;
    const float* a_ptr = a.data();
    const float* b_ptr = b.data();
    
    // Offload reduction computation to GPU
    // map(to:) - transfer input vectors to device
    // map(tofrom:) - transfer result bidirectionally
    // reduction(+:result) - parallel reduction on GPU
    #pragma omp target teams distribute parallel for \
        device(0) \
        map(to: a_ptr[0:size], b_ptr[0:size]) \
        map(tofrom: result) \
        reduction(+:result)
    for (size_t i = 0; i < size; i++) {
        result += a_ptr[i] * b_ptr[i];
    }
    
    return result;
}

// Matrix-vector multiplication using GPU offloading
// Demonstrates 2D data mapping and nested parallelism
void matrix_vector_multiply(const std::vector<float>& matrix,
                           const std::vector<float>& vector,
                           std::vector<float>& result,
                           size_t rows, size_t cols) {
    
    const float* m_ptr = matrix.data();
    const float* v_ptr = vector.data();
    float* r_ptr = result.data();
    
    // Transfer matrix and vector to device
    // Compute result on GPU and transfer back
    #pragma omp target teams distribute parallel for \
        device(0) \
        map(to: m_ptr[0:rows*cols], v_ptr[0:cols]) \
        map(from: r_ptr[0:rows])
    for (size_t i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            sum += m_ptr[i * cols + j] * v_ptr[j];
        }
        r_ptr[i] = sum;
    }
}

// Scalar multiplication using GPU offloading
// Simple element-wise operation demonstrating basic offloading pattern
void scalar_multiply(std::vector<float>& data,
                    float scalar,
                    size_t size) {
    
    float* data_ptr = data.data();
    
    // In-place operation on GPU
    // map(tofrom:) for bidirectional transfer
    #pragma omp target teams distribute parallel for \
        device(0) \
        map(tofrom: data_ptr[0:size])
    for (size_t i = 0; i < size; i++) {
        data_ptr[i] *= scalar;
    }
}

// Utility function to check if GPU is available
bool check_gpu_available() {
    int num_devices = omp_get_num_devices();
    std::cout << "Number of OpenMP target devices: " << num_devices << std::endl;
    return num_devices > 0;
}

// Benchmark function to compare GPU vs CPU execution
void benchmark_operations(size_t vector_size) {
    std::vector<float> a(vector_size, 1.0f);
    std::vector<float> b(vector_size, 2.0f);
    std::vector<float> result(vector_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    vector_add(a, b, result, vector_size);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "GPU Vector addition time: " << duration.count() << " microseconds" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    float dp_result = dot_product(a, b, vector_size);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "GPU Dot product time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Dot product result: " << dp_result << std::endl;
}

// Main demonstration function
int main() {
    std::cout << "OpenMP GPU Offloading Vector Operations" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (!check_gpu_available()) {
        std::cerr << "Warning: No GPU devices detected!" << std::endl;
        return 1;
    }
    
    const size_t size = 1000000;
    benchmark_operations(size);
    
    return 0;
}