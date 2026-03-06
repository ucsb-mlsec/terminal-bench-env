#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

#define NUM_THREADS 4
#define ITERATIONS 100000000

uint64_t counters[NUM_THREADS];

void* increment_counter(void* arg) {
    int thread_id = *(int*)arg;
    for (uint64_t i = 0; i < ITERATIONS; i++) {
        counters[thread_id]++;
    }
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];
    struct timespec start, end;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        counters[i] = 0;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, increment_counter, &thread_ids[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    printf("Time: %.3f seconds\n", elapsed);
    
    for (int i = 0; i < NUM_THREADS; i++) {
        printf("Counter[%d]: %lu\n", i, counters[i]);
    }
    
    return 0;
}