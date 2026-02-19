#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define BINS 100
#define NUM_THREADS 4
#define DATA_SIZE 1000000

int main() {
    int histogram[BINS] = {0};
    int *data;
    int i, value;
    FILE *file;
    double start_time, end_time, elapsed_time;
    
    // Allocate memory for data
    data = (int *)malloc(DATA_SIZE * sizeof(int));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Read data from file
    file = fopen("data.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Cannot open data.txt\n");
        free(data);
        return 1;
    }
    
    for (i = 0; i < DATA_SIZE; i++) {
        if (fscanf(file, "%d", &data[i]) != 1) {
            fprintf(stderr, "Error reading data at position %d\n", i);
            fclose(file);
            free(data);
            return 1;
        }
    }
    fclose(file);
    
    // Set number of threads
    omp_set_num_threads(NUM_THREADS);
    
    // Start timing
    start_time = omp_get_wtime();
    
    // Parallel histogram computation with atomic operations
    // This causes performance problems due to high contention
    #pragma omp parallel for private(value) shared(histogram, data)
    for (i = 0; i < DATA_SIZE; i++) {
        value = data[i];
        #pragma omp atomic
        histogram[value]++;
    }
    
    // End timing
    end_time = omp_get_wtime();
    elapsed_time = end_time - start_time;
    
    // Print timing results
    printf("Histogram computation time: %.3f seconds\n", elapsed_time);
    
    // Print first 10 bins for verification
    printf("\nFirst 10 bins:\n");
    for (i = 0; i < 10; i++) {
        printf("Bin %d: %d\n", i, histogram[i]);
    }
    
    // Verify total count
    int total = 0;
    for (i = 0; i < BINS; i++) {
        total += histogram[i];
    }
    printf("\nTotal count: %d (expected: %d)\n", total, DATA_SIZE);
    
    // Free allocated memory
    free(data);
    
    return 0;
}