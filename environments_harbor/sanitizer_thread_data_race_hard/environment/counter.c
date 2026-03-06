#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

#define NUM_THREADS 10
#define ITERATIONS 100000

int counter = 0;

void* increment_counter(void* arg) {
    for (int i = 0; i < ITERATIONS; i++) {
        counter++;
    }
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    
    printf("Starting counter test with %d threads, %d iterations each\n", 
           NUM_THREADS, ITERATIONS);
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, increment_counter, NULL) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            return 1;
        }
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Final counter value: %d\n", counter);
    printf("Expected value: %d\n", NUM_THREADS * ITERATIONS);
    
    if (counter == NUM_THREADS * ITERATIONS) {
        printf("SUCCESS: Counter matches expected value\n");
    } else {
        printf("FAILURE: Counter does not match expected value\n");
    }
    
    return 0;
}