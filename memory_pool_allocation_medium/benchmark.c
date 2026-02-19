#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

// External interface functions to be implemented in pool_allocator.c
extern void* pool_init(size_t block_size, size_t num_blocks);
extern void* pool_alloc(void* pool);
extern void pool_free(void* pool, void* ptr);
extern void pool_destroy(void* pool);

#define BLOCK_SIZE 64
#define NUM_BLOCKS 10000
#define FIRST_ALLOC 100000
#define SECOND_ALLOC 50000
#define TOTAL_ALLOC 150000

int main() {
    void* pool;
    void* pointers[TOTAL_ALLOC];
    clock_t start_time, end_time;
    double elapsed_ms;
    double allocs_per_sec;
    bool success = true;
    int i;
    
    printf("Initializing memory pool allocator benchmark...\n");
    printf("Block size: %d bytes\n", BLOCK_SIZE);
    printf("Pool capacity: %d blocks\n", NUM_BLOCKS);
    printf("\n");
    
    // Initialize the pool
    pool = pool_init(BLOCK_SIZE, NUM_BLOCKS);
    if (pool == NULL) {
        fprintf(stderr, "ERROR: Failed to initialize pool\n");
        return 1;
    }
    
    printf("Pool initialized successfully\n");
    
    // Start timing
    start_time = clock();
    
    // Phase 1: Perform 100,000 allocations
    printf("Phase 1: Allocating %d blocks...\n", FIRST_ALLOC);
    for (i = 0; i < FIRST_ALLOC; i++) {
        pointers[i] = pool_alloc(pool);
        if (pointers[i] == NULL) {
            fprintf(stderr, "ERROR: Allocation failed at index %d\n", i);
            success = false;
            break;
        }
        // Write to allocated memory to ensure it's valid
        *((char*)pointers[i]) = (char)(i & 0xFF);
    }
    
    if (!success) {
        pool_destroy(pool);
        return 1;
    }
    
    printf("Phase 1 completed: %d blocks allocated\n", FIRST_ALLOC);
    
    // Phase 2: Randomly deallocate 50% of allocated blocks
    printf("Phase 2: Deallocating 50%% of blocks...\n");
    int dealloc_count = 0;
    for (i = 0; i < FIRST_ALLOC; i += 2) {
        pool_free(pool, pointers[i]);
        pointers[i] = NULL;
        dealloc_count++;
    }
    printf("Phase 2 completed: %d blocks deallocated\n", dealloc_count);
    
    // Phase 3: Perform another 50,000 allocations (reusing freed blocks)
    printf("Phase 3: Allocating %d more blocks (reusing freed blocks)...\n", SECOND_ALLOC);
    for (i = FIRST_ALLOC; i < TOTAL_ALLOC; i++) {
        pointers[i] = pool_alloc(pool);
        if (pointers[i] == NULL) {
            fprintf(stderr, "ERROR: Allocation failed at index %d in phase 3\n", i);
            success = false;
            break;
        }
        // Write to allocated memory to ensure it's valid
        *((char*)pointers[i]) = (char)(i & 0xFF);
    }
    
    if (!success) {
        pool_destroy(pool);
        return 1;
    }
    
    printf("Phase 3 completed: %d blocks allocated\n", SECOND_ALLOC);
    
    // Phase 4: Deallocate all remaining blocks
    printf("Phase 4: Deallocating all remaining blocks...\n");
    int final_dealloc = 0;
    for (i = 0; i < TOTAL_ALLOC; i++) {
        if (pointers[i] != NULL) {
            pool_free(pool, pointers[i]);
            pointers[i] = NULL;
            final_dealloc++;
        }
    }
    printf("Phase 4 completed: %d blocks deallocated\n", final_dealloc);
    
    // Stop timing
    end_time = clock();
    
    // Destroy the pool
    pool_destroy(pool);
    printf("Pool destroyed successfully\n");
    
    // Calculate timing statistics
    elapsed_ms = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    allocs_per_sec = (TOTAL_ALLOC / (elapsed_ms / 1000.0));
    
    printf("\n");
    printf("=== BENCHMARK RESULTS ===\n");
    printf("Total allocations: %d\n", TOTAL_ALLOC);
    printf("Execution time: %.2f ms\n", elapsed_ms);
    printf("Allocations per second: %.0f\n", allocs_per_sec);
    printf("Status: %s\n", success ? "SUCCESS" : "FAILURE");
    printf("=========================\n");
    
    return success ? 0 : 1;
}