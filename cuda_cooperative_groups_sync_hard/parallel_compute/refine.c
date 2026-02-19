#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

#define MAX_ITERATIONS 10
#define CONVERGENCE_THRESHOLD 0.0001

typedef struct {
    double *data;
    double *temp_data;
    int start_idx;
    int end_idx;
    int size;
    int thread_id;
    int num_threads;
} ThreadData;

// Global variables - shared among threads
double *global_data = NULL;
double *temp_buffer = NULL;
int data_size = 0;
pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t iteration_barrier;

// Statistics for tracking convergence
double max_change = 0.0;
int converged_count = 0;

// Function to read data from input file
int read_input_file(const char *filename, double **data) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open input file %s\n", filename);
        return -1;
    }

    // Count lines
    int count = 0;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp)) {
        count++;
    }
    rewind(fp);

    // Allocate memory
    *data = (double *)malloc(count * sizeof(double));
    if (!*data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return -1;
    }

    // Read values
    for (int i = 0; i < count; i++) {
        if (fscanf(fp, "%lf", &(*data)[i]) != 1) {
            fprintf(stderr, "Error: Failed to read value at line %d\n", i + 1);
            free(*data);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return count;
}

// Function to write data to output file
int write_output_file(const char *filename, double *data, int size) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open output file %s\n", filename);
        return -1;
    }

    for (int i = 0; i < size; i++) {
        fprintf(fp, "%.6f\n", data[i]);
    }

    fclose(fp);
    return 0;
}

// Refinement computation - BUG 1: Missing mutex for max_change update
void compute_local_refinement(ThreadData *td, int iteration) {
    double local_max_change = 0.0;
    
    for (int i = td->start_idx; i < td->end_idx; i++) {
        double old_value = td->data[i];
        double new_value = old_value;
        
        // Apply refinement based on neighbors
        if (i > 0 && i < td->size - 1) {
            // Average with neighbors and apply correction factor
            double avg = (td->data[i-1] + td->data[i] + td->data[i+1]) / 3.0;
            double correction = (avg - old_value) * 0.3;
            new_value = old_value + correction;
            
            // Add iteration-dependent refinement
            new_value += 0.001 * sin(iteration * 0.5);
        } else if (i == 0 && td->size > 1) {
            new_value = (td->data[i] + td->data[i+1]) / 2.0;
        } else if (i == td->size - 1 && td->size > 1) {
            new_value = (td->data[i-1] + td->data[i]) / 2.0;
        }
        
        td->temp_data[i] = new_value;
        
        double change = fabs(new_value - old_value);
        if (change > local_max_change) {
            local_max_change = change;
        }
    }
    
    // BUG 1: Race condition - multiple threads updating max_change without lock
    if (local_max_change > max_change) {
        max_change = local_max_change;
    }
}

// Secondary refinement pass - smoothing
void apply_smoothing(ThreadData *td) {
    for (int i = td->start_idx; i < td->end_idx; i++) {
        if (i > 0 && i < td->size - 1) {
            // Weighted smoothing
            double smoothed = 0.25 * td->temp_data[i-1] + 
                            0.5 * td->temp_data[i] + 
                            0.25 * td->temp_data[i+1];
            td->data[i] = smoothed;
        } else {
            td->data[i] = td->temp_data[i];
        }
    }
}

// Thread worker function
void *refinement_worker(void *arg) {
    ThreadData *td = (ThreadData *)arg;
    
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Reset convergence tracking
        if (td->thread_id == 0) {
            max_change = 0.0;
            converged_count = 0;
        }
        
        // BUG 2: Missing barrier before computation starts
        // Threads may start computing before thread 0 resets shared variables
        
        // First refinement pass
        compute_local_refinement(td, iter);
        
        // BUG 3: Missing barrier between refinement and smoothing
        // Some threads may start smoothing while others are still in refinement
        
        // Smoothing pass
        apply_smoothing(td);
        
        // Wait for all threads to complete iteration
        pthread_barrier_wait(&iteration_barrier);
        
        // Check convergence
        if (max_change < CONVERGENCE_THRESHOLD) {
            // BUG 4: Race condition on converged_count
            converged_count++;
            if (converged_count == td->num_threads) {
                if (td->thread_id == 0) {
                    printf("Converged after %d iterations\n", iter + 1);
                }
                break;
            }
        }
        
        pthread_barrier_wait(&iteration_barrier);
    }
    
    return NULL;
}

// Additional computation for stress testing synchronization
void compute_statistics(ThreadData *td) {
    double local_sum = 0.0;
    double local_sum_sq = 0.0;
    
    for (int i = td->start_idx; i < td->end_idx; i++) {
        local_sum += td->data[i];
        local_sum_sq += td->data[i] * td->data[i];
    }
    
    // BUG 5: Missing mutex lock for shared statistics update
    pthread_mutex_lock(&stats_mutex);
    // Intentionally create a race by not protecting all shared access
    static double global_sum = 0.0;
    static double global_sum_sq = 0.0;
    pthread_mutex_unlock(&stats_mutex);
    
    // Race condition: updating without lock
    global_sum += local_sum;
    global_sum_sq += local_sum_sq;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <num_threads>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int num_threads = atoi(argv[3]);

    if (num_threads <= 0) {
        fprintf(stderr, "Error: Number of threads must be positive\n");
        return 1;
    }

    // Read input data
    data_size = read_input_file(input_file, &global_data);
    if (data_size <= 0) {
        return 1;
    }

    printf("Processing %d values with %d threads\n", data_size, num_threads);

    // Allocate temporary buffer
    temp_buffer = (double *)malloc(data_size * sizeof(double));
    if (!temp_buffer) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(global_data);
        return 1;
    }

    // Initialize barrier
    pthread_barrier_init(&iteration_barrier, NULL, num_threads);

    // Create threads
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ThreadData *thread_data = (ThreadData *)malloc(num_threads * sizeof(ThreadData));

    int chunk_size = data_size / num_threads;
    int remainder = data_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].data = global_data;
        thread_data[i].temp_data = temp_buffer;
        thread_data[i].size = data_size;
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        
        thread_data[i].start_idx = i * chunk_size + (i < remainder ? i : remainder);
        thread_data[i].end_idx = thread_data[i].start_idx + chunk_size + (i < remainder ? 1 : 0);

        if (pthread_create(&threads[i], NULL, refinement_worker, &thread_data[i]) != 0) {
            fprintf(stderr, "Error: Failed to create thread %d\n", i);
            free(global_data);
            free(temp_buffer);
            free(threads);
            free(thread_data);
            return 1;
        }
    }

    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Write output
    if (write_output_file(output_file, global_data, data_size) != 0) {
        free(global_data);
        free(temp_buffer);
        free(threads);
        free(thread_data);
        pthread_barrier_destroy(&iteration_barrier);
        return 1;
    }

    printf("Results written to %s\n", output_file);

    // Cleanup
    free(global_data);
    free(temp_buffer);
    free(threads);
    free(thread_data);
    pthread_barrier_destroy(&iteration_barrier);
    pthread_mutex_destroy(&stats_mutex);

    return 0;
}