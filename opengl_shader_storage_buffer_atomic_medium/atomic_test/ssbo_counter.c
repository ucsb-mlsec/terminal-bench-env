#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

// Simulation parameters matching typical SSBO usage patterns
#define NUM_THREADS 10
#define INCREMENTS_PER_THREAD 100
#define BUFFER_SIZE 16

// Structure representing an OpenGL SSBO with counter array
// In actual shader code, this would be:
// layout(std430, binding = 0) buffer CounterBuffer {
//     uint counters[16];
// };
typedef struct {
    int counters[BUFFER_SIZE];
} SSBOBuffer;

// Global shared buffer simulating GPU SSBO
SSBOBuffer buffer;

// Thread worker function simulating parallel shader invocations
// Each thread represents multiple shader invocations processing pixels
void* shader_invocation_worker(void* arg) {
    long thread_id = (long)arg;
    
    // Each "shader invocation" performs multiple atomic increments
    for (int i = 0; i < INCREMENTS_PER_THREAD; i++) {
        // Select counter slot based on thread ID (simulating different pixel regions)
        int slot = thread_id % BUFFER_SIZE;
        
        // BUG: This is NOT atomic! 
        // This performs read-modify-write as separate operations
        // Under concurrent access, multiple threads can read the same value,
        // increment it, and write back, causing lost updates
        //
        // In shader code, developers might mistakenly write:
        //   counters[slot] = counters[slot] + 1;
        // instead of:
        //   atomicAdd(counters[slot], 1);
        buffer.counters[slot] = buffer.counters[slot] + 1;
        
        // The correct atomic operation would be:
        // atomic_fetch_add(&buffer.counters[slot], 1);
    }
    
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    
    printf("SSBO Atomic Counter Simulation\n");
    printf("===============================\n");
    printf("Simulating %d parallel shader invocations\n", NUM_THREADS);
    printf("Each performing %d atomic increments\n", INCREMENTS_PER_THREAD);
    printf("Using %d counter slots in SSBO\n\n", BUFFER_SIZE);
    
    // Initialize buffer to zero (like clearing SSBO before dispatch)
    memset(&buffer, 0, sizeof(SSBOBuffer));
    
    // Launch threads to simulate parallel shader execution
    printf("Launching parallel threads...\n");
    for (long i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, shader_invocation_worker, (void*)i) != 0) {
            fprintf(stderr, "Error creating thread %ld\n", i);
            return 1;
        }
    }
    
    // Wait for all shader invocations to complete
    printf("Waiting for completion...\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Calculate expected total
    int expected_total = NUM_THREADS * INCREMENTS_PER_THREAD;
    
    // Sum actual values from all counter slots
    int actual_total = 0;
    printf("\nCounter slot values:\n");
    for (int i = 0; i < BUFFER_SIZE; i++) {
        printf("  Slot %2d: %d\n", i, buffer.counters[i]);
        actual_total += buffer.counters[i];
    }
    
    // Report results
    printf("\n=== RESULTS ===\n");
    printf("EXPECTED_TOTAL=%d\n", expected_total);
    printf("ACTUAL_TOTAL=%d\n", actual_total);
    
    if (actual_total == expected_total) {
        printf("VERIFICATION=PASS\n");
    } else {
        printf("VERIFICATION=FAIL\n");
        printf("\nLost %d increments due to race conditions!\n", expected_total - actual_total);
        printf("This demonstrates the importance of atomic operations in SSBO access.\n");
    }
    
    return 0;
}