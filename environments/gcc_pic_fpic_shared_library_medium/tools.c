#include <stdio.h>
#include <stdlib.h>
#include "tools.h"

// External function from libcalc.a
extern int add_numbers(int a, int b);
extern int multiply_numbers(int a, int b);

int calculate_result(int a, int b) {
    // Perform a simple calculation using both parameters
    int sum = add_numbers(a, b);
    int product = multiply_numbers(a, 2);
    return sum + product;
}

void process_data(int value) {
    // Print a message with the value
    printf("Processing data with value: %d\n", value);
    
    // Perform some additional processing
    int result = calculate_result(value, 10);
    printf("Calculated result: %d\n", result);
    
    // Add some validation
    if (value < 0) {
        printf("Warning: Negative value detected\n");
    } else if (value == 0) {
        printf("Info: Zero value provided\n");
    } else {
        printf("Info: Positive value %d is valid\n", value);
    }
}

const char* format_output(int value) {
    // Return a static formatted string
    static char buffer[256];
    snprintf(buffer, sizeof(buffer), "Formatted Output: [Value=%d, Status=OK]", value);
    return buffer;
}

int validate_input(int value) {
    // Simple validation function
    if (value >= 0 && value <= 1000) {
        return 1; // Valid
    }
    return 0; // Invalid
}

void print_statistics(int* data, int count) {
    // Helper function to print statistics
    if (data == NULL || count <= 0) {
        printf("No data to process\n");
        return;
    }
    
    int sum = 0;
    for (int i = 0; i < count; i++) {
        sum = add_numbers(sum, data[i]);
    }
    
    printf("Statistics: Count=%d, Sum=%d\n", count, sum);
}