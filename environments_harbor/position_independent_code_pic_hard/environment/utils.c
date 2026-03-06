#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Global variable that might cause issues with PIC
static int validation_counter = 0;

// Function pointer that might cause relocation issues
static int (*validator_func)(void*, size_t) = NULL;

int validate_input(void* data, size_t size) {
    validation_counter++;
    
    if (data == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to validate_input\n");
        return 0;
    }
    
    if (size <= 0) {
        fprintf(stderr, "Error: Invalid size %zu passed to validate_input\n", size);
        return 0;
    }
    
    return 1;
}

void* allocate_buffer(size_t size) {
    if (size == 0) {
        fprintf(stderr, "Error: Cannot allocate buffer of size 0\n");
        return NULL;
    }
    
    void* buffer = malloc(size);
    if (buffer == NULL) {
        fprintf(stderr, "Error: Failed to allocate %zu bytes\n", size);
        return NULL;
    }
    
    memset(buffer, 0, size);
    return buffer;
}

int format_output(const char* data, char* output, size_t output_size) {
    if (data == NULL || output == NULL) {
        fprintf(stderr, "Error: NULL pointer in format_output\n");
        return -1;
    }
    
    if (output_size == 0) {
        fprintf(stderr, "Error: Output buffer size is 0\n");
        return -1;
    }
    
    int written = snprintf(output, output_size, "[DATA: %s] (validation_count: %d)", 
                          data, validation_counter);
    
    if (written < 0) {
        fprintf(stderr, "Error: snprintf failed\n");
        return -1;
    }
    
    if ((size_t)written >= output_size) {
        fprintf(stderr, "Warning: Output was truncated\n");
        return -1;
    }
    
    return written;
}

void set_custom_validator(int (*func)(void*, size_t)) {
    validator_func = func;
}

int get_validation_count(void) {
    return validation_counter;
}

void reset_validation_count(void) {
    validation_counter = 0;
}