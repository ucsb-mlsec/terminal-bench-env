#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define NUM_TASKS 10000
#define NUM_THREADS 8
#define CALIBRATION_FACTOR 100

// Busy-wait function to simulate computational work
void busy_wait(long microseconds) {
    struct timeval start, current;
    gettimeofday(&start, NULL);
    
    volatile long dummy = 0;
    long elapsed = 0;
    
    while (elapsed < microseconds) {
        dummy++;
        gettimeofday(&current, NULL);
        elapsed = (current.tv_sec - start.tv_sec) * 1000000 + 
                  (current.tv_usec - start.tv_usec);
    }
}

// Convert cost units to microseconds
long calibrate_cost(int cost) {
    return cost * CALIBRATION_FACTOR;
}

int main() {
    int *task_costs;
    long *thread_times;
    int completed = 0;
    long max_time = 0, min_time = 0;
    
    // Allocate memory for task costs
    task_costs = (int *)malloc(NUM_TASKS * sizeof(int));
    if (task_costs == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for task_costs\n");
        return 1;
    }
    
    // Allocate memory for thread execution times
    thread_times = (long *)calloc(NUM_THREADS, sizeof(long));
    if (thread_times == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for thread_times\n");
        free(task_costs);
        return 1;
    }
    
    // Read task costs from file
    FILE *input_file = fopen("/workspace/task_costs.txt", "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error: Could not open task_costs.txt\n");
        free(task_costs);
        free(thread_times);
        return 1;
    }
    
    for (int i = 0; i < NUM_TASKS; i++) {
        if (fscanf(input_file, "%d", &task_costs[i]) != 1) {
            fprintf(stderr, "Error: Failed to read task cost at line %d\n", i + 1);
            fclose(input_file);
            free(task_costs);
            free(thread_times);
            return 1;
        }
    }
    fclose(input_file);
    
    // Set number of threads
    omp_set_num_threads(NUM_THREADS);
    
    // Process tasks with static scheduling (causes load imbalance)
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        struct timeval thread_start, thread_end;
        
        gettimeofday(&thread_start, NULL);
        
        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_TASKS; i++) {
            long work_time = calibrate_cost(task_costs[i]);
            busy_wait(work_time);
            
            #pragma omp atomic
            completed++;
        }
        
        gettimeofday(&thread_end, NULL);
        
        long thread_elapsed = (thread_end.tv_sec - thread_start.tv_sec) * 1000000 + 
                             (thread_end.tv_usec - thread_start.tv_usec);
        thread_times[thread_id] = thread_elapsed;
    }
    
    // Calculate max and min thread execution times
    max_time = thread_times[0];
    min_time = thread_times[0];
    
    for (int i = 1; i < NUM_THREADS; i++) {
        if (thread_times[i] > max_time) {
            max_time = thread_times[i];
        }
        if (thread_times[i] < min_time) {
            min_time = thread_times[i];
        }
    }
    
    // Write metrics to output file
    FILE *output_file = fopen("/workspace/metrics.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error: Could not open metrics.txt for writing\n");
        free(task_costs);
        free(thread_times);
        return 1;
    }
    
    fprintf(output_file, "completed=%d\n", completed);
    fprintf(output_file, "max_time=%ld\n", max_time);
    fprintf(output_file, "min_time=%ld\n", min_time);
    
    fclose(output_file);
    
    // Clean up
    free(task_costs);
    free(thread_times);
    
    return 0;
}