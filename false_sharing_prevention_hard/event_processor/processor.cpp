#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cstdint>

// Global counter array - one per thread
// These are stored consecutively in memory, causing false sharing
long long counters[8];

// Worker function that increments its own counter
void worker(int thread_id, long long num_iterations) {
    for (long long i = 0; i < num_iterations; i++) {
        counters[thread_id]++;
    }
}

int main(int argc, char* argv[]) {
    // Get number of threads from command line, default to 8
    int num_threads = 8;
    if (argc > 1) {
        num_threads = std::atoi(argv[1]);
        if (num_threads < 1 || num_threads > 8) {
            std::cerr << "Number of threads must be between 1 and 8" << std::endl;
            return 1;
        }
    }
    
    // Number of iterations per thread
    const long long num_iterations = 10000000;
    
    // Initialize all counters to 0
    for (int i = 0; i < 8; i++) {
        counters[i] = 0;
    }
    
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Create and launch threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.push_back(std::thread(worker, i, num_iterations));
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate elapsed time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Calculate sum of all counters to verify correctness
    long long total = 0;
    for (int i = 0; i < num_threads; i++) {
        total += counters[i];
    }
    
    // Print results
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Total events processed: " << total << std::endl;
    std::cout << "Expected: " << (num_threads * num_iterations) << std::endl;
    
    return 0;
}