#include <iostream>
#include <thread>
#include <vector>

// Global counter without any synchronization
int global_counter = 0;

// Function that increments the counter many times
void increment_counter(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        global_counter++;  // Race condition: non-atomic read-modify-write
    }
}

// Function that performs multiple operations on the counter
void perform_operations(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        int temp = global_counter;  // Read
        temp = temp + 1;             // Modify
        global_counter = temp;       // Write - race condition
    }
}

int main() {
    const int num_threads = 8;
    const int iterations_per_thread = 50000;
    
    std::vector<std::thread> threads;
    
    std::cout << "Starting counter service with " << num_threads << " threads\n";
    std::cout << "Each thread will perform " << iterations_per_thread << " increments\n";
    std::cout << "Expected final value: " << (num_threads * iterations_per_thread) << "\n\n";
    
    // Create and start threads
    for (int i = 0; i < num_threads; ++i) {
        if (i % 2 == 0) {
            threads.emplace_back(increment_counter, iterations_per_thread);
        } else {
            threads.emplace_back(perform_operations, iterations_per_thread);
        }
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Print final result
    std::cout << "Final counter value: " << global_counter << "\n";
    std::cout << "Expected value: " << (num_threads * iterations_per_thread) << "\n";
    
    if (global_counter != num_threads * iterations_per_thread) {
        std::cout << "ERROR: Counter mismatch detected! Data inconsistency occurred.\n";
    } else {
        std::cout << "Counter matches expected value (race conditions may still exist).\n";
    }
    
    return 0;
}