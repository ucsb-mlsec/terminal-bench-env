#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#define NUM_NODES 100000

typedef struct SensorNode {
    double timestamp;
    int sensor_id;
    double measurement_value;
    struct SensorNode *next;
} SensorNode;

// Function to create a new sensor node
SensorNode* create_node(double timestamp, int sensor_id, double measurement_value) {
    SensorNode *node = (SensorNode*)malloc(sizeof(SensorNode));
    if (node == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    node->timestamp = timestamp;
    node->sensor_id = sensor_id;
    node->measurement_value = measurement_value;
    node->next = NULL;
    return node;
}

// Function to create linked list with sensor data
SensorNode* create_sensor_list(int num_nodes) {
    SensorNode *head = NULL;
    SensorNode *tail = NULL;
    
    for (int i = 0; i < num_nodes; i++) {
        double timestamp = i * 0.1;
        int sensor_id = (i % 10) + 1;
        double measurement_value = 20.0 + ((double)(i % 100) / 10.0);
        
        SensorNode *node = create_node(timestamp, sensor_id, measurement_value);
        
        if (head == NULL) {
            head = node;
            tail = node;
        } else {
            tail->next = node;
            tail = node;
        }
    }
    
    return head;
}

// Function to compute statistics by traversing the list
void compute_statistics(SensorNode *head, double *sum, double *avg, double *min, double *max, int *count) {
    *sum = 0.0;
    *min = DBL_MAX;
    *max = -DBL_MAX;
    *count = 0;
    
    SensorNode *current = head;
    while (current != NULL) {
        double value = current->measurement_value;
        *sum += value;
        
        if (value < *min) {
            *min = value;
        }
        if (value > *max) {
            *max = value;
        }
        
        (*count)++;
        current = current->next;
    }
    
    if (*count > 0) {
        *avg = *sum / *count;
    } else {
        *avg = 0.0;
    }
}

// Function to free the linked list
void free_list(SensorNode *head) {
    SensorNode *current = head;
    while (current != NULL) {
        SensorNode *temp = current;
        current = current->next;
        free(temp);
    }
}

// Function to get time in milliseconds
long long get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)(ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL);
}

int main() {
    printf("Creating linked list with %d sensor nodes...\n", NUM_NODES);
    
    // Create the linked list
    SensorNode *head = create_sensor_list(NUM_NODES);
    
    printf("Computing statistics...\n");
    
    // Measure execution time for statistics computation
    long long start_time = get_time_ms();
    
    double sum, avg, min, max;
    int count;
    compute_statistics(head, &sum, &avg, &min, &max, &count);
    
    long long end_time = get_time_ms();
    long long elapsed_ms = end_time - start_time;
    
    // Print results
    printf("\n=== Results ===\n");
    printf("Execution Time: %lld MS\n", elapsed_ms);
    printf("Nodes processed: %d\n", count);
    printf("Sum: %.2f\n", sum);
    printf("Average: %.6f\n", avg);
    printf("Min: %.2f\n", min);
    printf("Max: %.2f\n", max);
    
    // Cleanup
    free_list(head);
    
    return 0;
}