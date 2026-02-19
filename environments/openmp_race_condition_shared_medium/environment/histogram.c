#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_BINS 10
#define MIN_VALUE 0.0
#define MAX_VALUE 100.0
#define BIN_WIDTH 10.0

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <num_threads>\n", argv[0]);
        return 1;
    }

    char *input_filename = argv[1];
    int num_threads = atoi(argv[2]);

    if (num_threads <= 0) {
        fprintf(stderr, "Error: Number of threads must be positive\n");
        return 1;
    }

    omp_set_num_threads(num_threads);

    // Open input file
    FILE *fp = fopen(input_filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", input_filename);
        return 1;
    }

    // Count number of values in file
    int num_values = 0;
    float temp;
    while (fscanf(fp, "%f", &temp) == 1) {
        num_values++;
    }
    rewind(fp);

    // Allocate array for data
    float *data = (float *)malloc(num_values * sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return 1;
    }

    // Read all values
    for (int i = 0; i < num_values; i++) {
        if (fscanf(fp, "%f", &data[i]) != 1) {
            fprintf(stderr, "Error: Failed to read value %d\n", i);
            free(data);
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);

    // Initialize histogram bins to zero
    int histogram[NUM_BINS];
    for (int i = 0; i < NUM_BINS; i++) {
        histogram[i] = 0;
    }

    // Process data in parallel and compute histogram
    // WARNING: This contains a race condition!
    #pragma omp parallel for
    for (int i = 0; i < num_values; i++) {
        float value = data[i];
        
        // Skip values outside the valid range
        if (value < MIN_VALUE || value >= MAX_VALUE) {
            continue;
        }

        // Calculate bin index
        int bin = (int)(value / BIN_WIDTH);
        
        // Ensure bin is within valid range
        if (bin >= 0 && bin < NUM_BINS) {
            // RACE CONDITION: Multiple threads may update the same bin simultaneously
            histogram[bin]++;
        }
    }

    // Print histogram results
    printf("Histogram Results:\n");
    printf("==================\n");
    int total_count = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        float bin_start = i * BIN_WIDTH;
        float bin_end = (i + 1) * BIN_WIDTH;
        printf("Bin %d [%.1f - %.1f): %d\n", i, bin_start, bin_end, histogram[i]);
        total_count += histogram[i];
    }
    printf("==================\n");
    printf("Total values counted: %d\n", total_count);
    printf("Total values processed: %d\n", num_values);

    // Clean up
    free(data);

    return 0;
}