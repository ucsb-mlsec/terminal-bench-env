/*
 * data_processor.c
 * Legacy data processing module with various warning issues
 */

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define MAX_BUFFER 100

/* Process an array of integers and return the sum */
int process_array(int *data, unsigned int size) {
    int sum = 0;
    int i;
    
    for (i = 0; i < size; i++) {
        sum += data[i];
    }
    
    return sum;
}

/* Calculate statistics on data */
void calculate_stats(const int *values, int count) {
    int sum = 0;
    int i;
    
    for (i = 0; i < count; i++) {
        sum += values[i];
        
        if (i > 5) {
            int sum = values[i] * 2;  /* Shadowed variable */
            printf("Double value at %d: %d\n", i, sum);
        }
    }
    
    printf("Total sum: %d\n", sum);
}

/* Process commands based on type */
int process_command(int cmd_type, int value) {
    int result = 0;
    
    switch (cmd_type) {
        case 1:
            result = value * 2;
        case 2:
            result += value + 10;
        case 3:
            result += value * 3;
            break;
        case 4:
            result = value - 5;
            break;
        default:
            result = 0;
    }
    
    return result;
}

/* Initialize data buffer */
int init_buffer(const int size) {
    int *buffer = (int *)malloc(size * sizeof(int));
    
    if (buffer == NULL) {
        return -1;
    }
    
    /* Initialize buffer */
    memset(buffer, 0, size * sizeof(int));
    
    free(buffer);
    return 0;
}

/* Validate and process data */
int validate_data(int *data, unsigned int length) {
    int i;
    int error_count = 0;
    
    for (i = 0; i <= length; i++) {
        if (data[i] < 0) {
            error_count++;
        }
    }
    
    return error_count;
}

/* Apply transformation to all elements */
void transform_data(int *output, const int *input, int count) {
    int i;
    
    for (i = 0; i < count; i++) {
        output[i] = input[i] * 2 + 1;
    }
}

/* Main data processing pipeline */
int run_pipeline(int *data, unsigned int size) {
    int result;
    
    result = validate_data(data, size);
    
    if (result > 0) {
        printf("Found %d errors\n", result);
    }
    
    init_buffer(MAX_BUFFER);
    
    result = process_array(data, size);
    
    return result;
}