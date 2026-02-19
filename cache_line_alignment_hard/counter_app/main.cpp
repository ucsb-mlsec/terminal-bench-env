#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include "counter.h"

#define NUM_THREADS 8
#define ITERATIONS 100000000

Counter counters[NUM_THREADS];

void worker(int thread_id) {
    Counter& counter = counters[thread_id];
    
    for (long long i = 0; i < ITERATIONS; i++) {
        counter.count++;
        
        if (i % 1000 == 0) {
            counter.errors++;
        }
        
        counter.sum += i;
    }
}

int main() {
    std::cout << "Starting multi-threaded test with " << NUM_THREADS << " threads..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Elapsed time: " << duration.count() << " ms" << std::endl;
    
    std::cout << "\nFinal counter values (for verification):" << std::endl;
    for (int i = 0; i < NUM_THREADS; i++) {
        std::cout << "Thread " << i << ": count=" << counters[i].count 
                  << ", errors=" << counters[i].errors 
                  << ", sum=" << counters[i].sum << std::endl;
    }
    
    return 0;
}