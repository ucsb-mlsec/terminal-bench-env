#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

#define MEMORY_SIZE (1024 * 1024 * 1024)  // 1GB
#define ITERATIONS 15
#define PAGE_SIZE 4096

int main() {
    struct timespec start, end;
    long long elapsed_ns;
    int elapsed_ms;
    
    // Record start time
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Allocate large memory region using mmap for better control
    void *buffer = mmap(NULL, MEMORY_SIZE, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (buffer == MAP_FAILED) {
        perror("mmap failed");
        return 1;
    }
    
    // Initialize buffer to ensure pages are allocated
    memset(buffer, 0, MEMORY_SIZE);
    
    volatile long long sum = 0;
    
    // Perform memory-intensive operations with stride access pattern
    // This will stress the TLB by accessing many different pages
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Access every page in the buffer with a stride pattern
        for (size_t offset = 0; offset < MEMORY_SIZE; offset += PAGE_SIZE) {
            // Read and modify value at this offset
            char *ptr = (char *)buffer + offset;
            sum += *ptr;
            *ptr = (char)(sum & 0xFF);
        }
        
        // Additional random-like access pattern within each iteration
        for (size_t i = 0; i < MEMORY_SIZE / (PAGE_SIZE * 4); i++) {
            size_t offset = ((i * 37 + iter * 97) % (MEMORY_SIZE / PAGE_SIZE)) * PAGE_SIZE;
            char *ptr = (char *)buffer + offset;
            sum += *ptr;
            *ptr = (char)((sum + i) & 0xFF);
        }
    }
    
    // Record end time
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // Calculate elapsed time in milliseconds
    elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    elapsed_ms = (int)(elapsed_ns / 1000000);
    
    // Print execution time
    printf("Execution time: %d ms\n", elapsed_ms);
    
    // Prevent optimization from removing our computation
    if (sum == 0x123456789ABCDEFLL) {
        printf("Unexpected sum\n");
    }
    
    // Free allocated memory
    munmap(buffer, MEMORY_SIZE);
    
    return 0;
}