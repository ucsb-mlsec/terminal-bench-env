#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cmath>

// Function to compute sum of a portion of the vector
void partial_sum(const std::vector<double>& data, size_t start, size_t end, double& result) {
    result = 0.0;
    for (size_t i = start; i < end; ++i) {
        result += std::sqrt(data[i] * data[i] + 1.0);
    }
}

// Function to compute squared values in parallel
void parallel_square(std::vector<double>& data, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
        data[i] = data[i] * data[i];
    }
}

int main() {
    const size_t DATA_SIZE = 10000000;
    const unsigned int NUM_THREADS = std::thread::hardware_concurrency();
    
    std::cout << "Parallel Compute Engine" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Data size: " << DATA_SIZE << std::endl;
    std::cout << "Number of threads: " << NUM_THREADS << std::endl;
    
    // Initialize data
    std::vector<double> data(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; ++i) {
        data[i] = static_cast<double>(i % 1000) + 0.5;
    }
    
    // Parallel sum computation
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    std::vector<double> partial_results(NUM_THREADS);
    size_t chunk_size = DATA_SIZE / NUM_THREADS;
    
    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == NUM_THREADS - 1) ? DATA_SIZE : (i + 1) * chunk_size;
        threads.emplace_back(partial_sum, std::cref(data), start, end, std::ref(partial_results[i]));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Combine results
    double total_sum = std::accumulate(partial_results.begin(), partial_results.end(), 0.0);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\nParallel sum result: " << total_sum << std::endl;
    std::cout << "Computation time: " << duration.count() << " ms" << std::endl;
    
    // Parallel transformation
    std::cout << "\nPerforming parallel square transformation..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    threads.clear();
    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == NUM_THREADS - 1) ? DATA_SIZE : (i + 1) * chunk_size;
        threads.emplace_back(parallel_square, std::ref(data), start, end);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double average = std::accumulate(data.begin(), data.begin() + 100, 0.0) / 100.0;
    std::cout << "Transformation complete. Average of first 100 elements: " << average << std::endl;
    std::cout << "Transformation time: " << duration.count() << " ms" << std::endl;
    
    std::cout << "\nComputation finished successfully!" << std::endl;
    
    return 0;
}