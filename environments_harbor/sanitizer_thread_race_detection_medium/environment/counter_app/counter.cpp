#include <iostream>
#include <thread>
#include <vector>

int counter = 0;

void incrementCounter(int iterations) {
    for (int i = 0; i < iterations; i++) {
        counter++;
    }
}

int main() {
    const int NUM_THREADS = 10;
    const int INCREMENTS_PER_THREAD = 1000;
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.push_back(std::thread(incrementCounter, INCREMENTS_PER_THREAD));
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "Final counter value: " << counter << std::endl;
    std::cout << "Expected value: " << (NUM_THREADS * INCREMENTS_PER_THREAD) << std::endl;
    
    return 0;
}