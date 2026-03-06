#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define DATA_SIZE 10000
#define ITERATIONS 100

int main() {
    double *sensor_data;
    double *processed_data;
    double global_sum = 0.0;
    double global_sum_sq = 0.0;
    double mean, variance;
    int i, iter;
    FILE *fp;
    double start_time, end_time;
    
    // Allocate memory for sensor data
    sensor_data = (double *)malloc(DATA_SIZE * sizeof(double));
    processed_data = (double *)malloc(DATA_SIZE * sizeof(double));
    
    if (sensor_data == NULL || processed_data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Read binary sensor data
    fp = fopen("/workspace/sensor_analysis/data.bin", "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error opening data file\n");
        free(sensor_data);
        free(processed_data);
        return 1;
    }
    
    size_t read_count = fread(sensor_data, sizeof(double), DATA_SIZE, fp);
    fclose(fp);
    
    if (read_count != DATA_SIZE) {
        fprintf(stderr, "Error reading data: expected %d, got %zu\n", DATA_SIZE, read_count);
        free(sensor_data);
        free(processed_data);
        return 1;
    }
    
    // Initialize processed data
    for (i = 0; i < DATA_SIZE; i++) {
        processed_data[i] = sensor_data[i];
    }
    
    printf("Starting sensor data processing with %d threads...\n", omp_get_max_threads());
    start_time = omp_get_wtime();
    
    // Iterative processing with DELIBERATE PERFORMANCE BOTTLENECK
    // Using excessive synchronization inside the parallel loop
    for (iter = 0; iter < ITERATIONS; iter++) {
        global_sum = 0.0;
        global_sum_sq = 0.0;
        
        // Parallel loop with CRITICAL SECTION inside - causes serialization!
        #pragma omp parallel for
        for (i = 0; i < DATA_SIZE; i++) {
            double value = processed_data[i];
            double adjusted_value;
            
            // Some computation
            adjusted_value = value * 1.001 + 0.01 * sin(value);
            
            // PERFORMANCE BOTTLENECK: Critical section inside tight loop
            // This forces threads to execute serially when updating shared variables
            #pragma omp critical
            {
                global_sum += adjusted_value;
                global_sum_sq += adjusted_value * adjusted_value;
            }
            
            processed_data[i] = adjusted_value;
        }
        
        // Compute statistics for this iteration
        mean = global_sum / DATA_SIZE;
        variance = (global_sum_sq / DATA_SIZE) - (mean * mean);
        
        // Apply correction based on statistics
        #pragma omp parallel for
        for (i = 0; i < DATA_SIZE; i++) {
            processed_data[i] = processed_data[i] - mean * 0.0001;
        }
    }
    
    end_time = omp_get_wtime();
    
    // Final statistics computation
    global_sum = 0.0;
    global_sum_sq = 0.0;
    
    for (i = 0; i < DATA_SIZE; i++) {
        global_sum += processed_data[i];
        global_sum_sq += processed_data[i] * processed_data[i];
    }
    
    mean = global_sum / DATA_SIZE;
    variance = (global_sum_sq / DATA_SIZE) - (mean * mean);
    
    // Output results
    printf("Processing complete!\n");
    printf("Execution time: %.4f seconds\n", end_time - start_time);
    printf("Final mean: %.10f\n", mean);
    printf("Final variance: %.10f\n", variance);
    printf("Sample values: %.6f %.6f %.6f\n", 
           processed_data[0], processed_data[DATA_SIZE/2], processed_data[DATA_SIZE-1]);
    
    // Cleanup
    free(sensor_data);
    free(processed_data);
    
    return 0;
}