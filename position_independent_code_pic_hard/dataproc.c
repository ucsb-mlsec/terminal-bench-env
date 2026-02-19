#include <stdio.h>
#include <stdlib.h>

// Global variables that may cause relocation issues
static int processing_mode = 1;
static double scaling_factor = 2.0;
static int statistics_cache[100];

// Static helper function for internal use
static int validate_input(int *data, int size) {
    if (data == NULL || size <= 0) {
        return -1;
    }
    return 0;
}

// Static helper function for bounds checking
static int check_bounds(int value, int min, int max) {
    return (value >= min && value <= max);
}

// Static helper for calculating sum
static long calculate_sum(int *array, int size) {
    long sum = 0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}

// Exported function: Transform data by multiplying each element
int transform_data(int *input, int size, int *output) {
    if (validate_input(input, size) != 0) {
        fprintf(stderr, "Invalid input to transform_data\n");
        return -1;
    }
    
    if (output == NULL) {
        return -1;
    }
    
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * (int)scaling_factor;
    }
    
    return 0;
}

// Exported function: Process array with a specific operation
int process_array(int *data, int size, int operation) {
    if (validate_input(data, size) != 0) {
        return -1;
    }
    
    switch (operation) {
        case 0: // Increment all values
            for (int i = 0; i < size; i++) {
                data[i] += processing_mode;
            }
            break;
        case 1: // Double all values
            for (int i = 0; i < size; i++) {
                data[i] *= 2;
            }
            break;
        case 2: // Absolute value
            for (int i = 0; i < size; i++) {
                if (data[i] < 0) {
                    data[i] = -data[i];
                }
            }
            break;
        default:
            return -1;
    }
    
    return 0;
}

// Exported function: Filter out negative values
int filter_values(int *input, int size, int *output, int *output_size) {
    if (validate_input(input, size) != 0) {
        return -1;
    }
    
    if (output == NULL || output_size == NULL) {
        return -1;
    }
    
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (input[i] >= 0 && check_bounds(input[i], 0, 1000000)) {
            output[count] = input[i];
            count++;
        }
    }
    
    *output_size = count;
    return 0;
}

// Exported function: Calculate statistics on array
int aggregate_stats(int *data, int size, double *sum, double *average, int *min, int *max) {
    if (validate_input(data, size) != 0) {
        return -1;
    }
    
    if (sum == NULL || average == NULL || min == NULL || max == NULL) {
        return -1;
    }
    
    long total = calculate_sum(data, size);
    *sum = (double)total;
    *average = (double)total / size;
    
    *min = data[0];
    *max = data[0];
    
    for (int i = 1; i < size; i++) {
        if (data[i] < *min) {
            *min = data[i];
        }
        if (data[i] > *max) {
            *max = data[i];
        }
    }
    
    // Cache some statistics
    if (size <= 100) {
        for (int i = 0; i < size; i++) {
            statistics_cache[i] = data[i];
        }
    }
    
    return 0;
}

// Additional exported function for setting processing mode
int set_processing_mode(int mode) {
    if (mode < 0 || mode > 10) {
        return -1;
    }
    processing_mode = mode;
    return 0;
}

// Additional exported function for setting scaling factor
int set_scaling_factor(double factor) {
    if (factor < 0.0) {
        return -1;
    }
    scaling_factor = factor;
    return 0;
}