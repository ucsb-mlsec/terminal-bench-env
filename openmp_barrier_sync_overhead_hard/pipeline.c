#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DATA_SIZE 8000
#define NUM_BINS 100
#define BIN_SIZE 80

int main() {
    double *data = (double *)malloc(DATA_SIZE * sizeof(double));
    double *filtered = (double *)malloc(DATA_SIZE * sizeof(double));
    double *transformed = (double *)malloc(DATA_SIZE * sizeof(double));
    double *aggregated = (double *)malloc(NUM_BINS * sizeof(double));
    
    if (!data || !filtered || !transformed || !aggregated) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Read input data
    FILE *input_file = fopen("/workspace/sensor_data.txt", "r");
    if (!input_file) {
        fprintf(stderr, "Cannot open input file\n");
        free(data);
        free(filtered);
        free(transformed);
        free(aggregated);
        return 1;
    }
    
    for (int i = 0; i < DATA_SIZE; i++) {
        if (fscanf(input_file, "%lf", &data[i]) != 1) {
            fprintf(stderr, "Error reading data at position %d\n", i);
            fclose(input_file);
            free(data);
            free(filtered);
            free(transformed);
            free(aggregated);
            return 1;
        }
    }
    fclose(input_file);
    
    // Set number of threads
    omp_set_num_threads(8);
    
    // Start timing
    double start_time = omp_get_wtime();
    
    // Stage 1: Filtering
    #pragma omp parallel for
    for (int i = 0; i < DATA_SIZE; i++) {
        if (data[i] < 0) {
            filtered[i] = 0.0;
        } else {
            filtered[i] = data[i];
        }
    }
    // Implicit barrier here
    
    // Stage 2: Transformation
    #pragma omp parallel for
    for (int i = 0; i < DATA_SIZE; i++) {
        transformed[i] = sqrt(fabs(filtered[i])) * 1.5 + sin(filtered[i]);
    }
    // Implicit barrier here
    
    // Stage 3: Aggregation
    // Initialize aggregated array
    for (int i = 0; i < NUM_BINS; i++) {
        aggregated[i] = 0.0;
    }
    
    #pragma omp parallel for
    for (int bin = 0; bin < NUM_BINS; bin++) {
        double sum = 0.0;
        int start_idx = bin * BIN_SIZE;
        int end_idx = start_idx + BIN_SIZE;
        
        for (int i = start_idx; i < end_idx; i++) {
            sum += transformed[i];
        }
        aggregated[bin] = sum;
    }
    // Implicit barrier here
    
    // End timing
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    
    // Write output
    FILE *output_file = fopen("/workspace/output.dat", "w");
    if (!output_file) {
        fprintf(stderr, "Cannot open output file\n");
        free(data);
        free(filtered);
        free(transformed);
        free(aggregated);
        return 1;
    }
    
    for (int i = 0; i < NUM_BINS; i++) {
        fprintf(output_file, "%.10f\n", aggregated[i]);
    }
    fclose(output_file);
    
    // Print execution time
    printf("Execution time: %.6f seconds\n", elapsed_time);
    
    // Cleanup
    free(data);
    free(filtered);
    free(transformed);
    free(aggregated);
    
    return 0;
}