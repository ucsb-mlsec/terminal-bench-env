#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include "lockfree_queue.h"

#define NUM_PRODUCERS 4
#define NUM_CONSUMERS 4
#define MESSAGES_PER_PRODUCER 10000

static queue_t* global_queue;
static int received_messages[NUM_PRODUCERS * MESSAGES_PER_PRODUCER];
static volatile bool terminate_consumers = false;
static pthread_mutex_t tracking_mutex = PTHREAD_MUTEX_INITIALIZER;

void* producer_thread(void* arg) {
    long thread_id = (long)arg;
    int start = thread_id * MESSAGES_PER_PRODUCER;
    int end = (thread_id + 1) * MESSAGES_PER_PRODUCER;
    
    for (int i = start; i < end; i++) {
        enqueue(global_queue, i);
    }
    
    printf("Producer %ld finished (enqueued %d-%d)\n", thread_id, start, end-1);
    return NULL;
}

void* consumer_thread(void* arg) {
    long thread_id = (long)arg;
    int messages_consumed = 0;
    
    while (!terminate_consumers || !is_empty(global_queue)) {
        int value;
        if (dequeue(global_queue, &value)) {
            pthread_mutex_lock(&tracking_mutex);
            if (value >= 0 && value < NUM_PRODUCERS * MESSAGES_PER_PRODUCER) {
                received_messages[value]++;
            }
            pthread_mutex_unlock(&tracking_mutex);
            messages_consumed++;
        } else {
            usleep(1000);
        }
    }
    
    printf("Consumer %ld finished (consumed %d messages)\n", thread_id, messages_consumed);
    return NULL;
}

int main() {
    pthread_t producers[NUM_PRODUCERS];
    pthread_t consumers[NUM_CONSUMERS];
    
    printf("=== Lock-Free Queue Test Harness ===\n");
    printf("Producers: %d, Consumers: %d\n", NUM_PRODUCERS, NUM_CONSUMERS);
    printf("Total messages: %d\n\n", NUM_PRODUCERS * MESSAGES_PER_PRODUCER);
    
    // Initialize tracking array
    for (int i = 0; i < NUM_PRODUCERS * MESSAGES_PER_PRODUCER; i++) {
        received_messages[i] = 0;
    }
    
    // Initialize queue
    global_queue = queue_create();
    if (!global_queue) {
        fprintf(stderr, "Failed to create queue\n");
        return 1;
    }
    
    // Start consumer threads
    for (long i = 0; i < NUM_CONSUMERS; i++) {
        if (pthread_create(&consumers[i], NULL, consumer_thread, (void*)i) != 0) {
            fprintf(stderr, "Failed to create consumer thread %ld\n", i);
            return 1;
        }
    }
    
    // Start producer threads
    for (long i = 0; i < NUM_PRODUCERS; i++) {
        if (pthread_create(&producers[i], NULL, producer_thread, (void*)i) != 0) {
            fprintf(stderr, "Failed to create producer thread %ld\n", i);
            return 1;
        }
    }
    
    // Wait for all producers to finish
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        pthread_join(producers[i], NULL);
    }
    printf("\nAll producers finished\n");
    
    // Give consumers time to drain the queue
    sleep(2);
    
    // Signal consumers to terminate
    terminate_consumers = true;
    
    // Wait for all consumers to finish
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        pthread_join(consumers[i], NULL);
    }
    printf("All consumers finished\n\n");
    
    // Validate results
    int missing_count = 0;
    int duplicate_count = 0;
    int total_received = 0;
    
    for (int i = 0; i < NUM_PRODUCERS * MESSAGES_PER_PRODUCER; i++) {
        if (received_messages[i] == 0) {
            missing_count++;
        } else if (received_messages[i] > 1) {
            duplicate_count += (received_messages[i] - 1);
        }
        total_received += received_messages[i];
    }
    
    printf("=== Test Results ===\n");
    printf("Expected messages: %d\n", NUM_PRODUCERS * MESSAGES_PER_PRODUCER);
    printf("Total received: %d\n", total_received);
    printf("Missing messages: %d\n", missing_count);
    printf("Duplicate messages: %d\n", duplicate_count);
    
    if (missing_count == 0 && duplicate_count == 0) {
        printf("\n*** TEST PASSED ***\n");
        queue_destroy(global_queue);
        return 0;
    } else {
        printf("\n*** TEST FAILED ***\n");
        queue_destroy(global_queue);
        return 1;
    }
}