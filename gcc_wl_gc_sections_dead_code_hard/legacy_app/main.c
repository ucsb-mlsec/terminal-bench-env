#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global data - some used, some unused
static int used_lookup_table[256];
static int unused_cache[1024];
static const char* used_messages[] = {
    "Initialization complete",
    "Processing data",
    "Validation successful",
    "Operation finished"
};
static const char* unused_error_codes[] = {
    "ERR001: Legacy error",
    "ERR002: Deprecated warning",
    "ERR003: Old system fault",
    "ERR004: Ancient exception"
};
static double unused_coefficients[500];

// Used functions - core application logic
int calculate_sum(int* array, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}

int validate_input(int value) {
    if (value < 0 || value > 1000) {
        return 0;
    }
    return 1;
}

void process_data(int* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] * 2 + 1;
    }
}

int find_maximum(int* array, int size) {
    int max = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max) {
            max = array[i];
        }
    }
    return max;
}

void initialize_lookup_table(void) {
    for (int i = 0; i < 256; i++) {
        used_lookup_table[i] = i * i;
    }
}

int compute_average(int* array, int size) {
    int sum = calculate_sum(array, size);
    return sum / size;
}

void print_status(const char* message) {
    printf("[STATUS] %s\n", message);
}

// Unused functions - dead code that should be eliminated
void old_legacy_function(int param) {
    printf("This function is never called: %d\n", param);
    for (int i = 0; i < 100; i++) {
        param += i * 2;
    }
}

int deprecated_handler(char* buffer, int size) {
    int result = 0;
    for (int i = 0; i < size; i++) {
        result += buffer[i];
    }
    return result * 3;
}

void unused_calculator(double x, double y) {
    double result = x * y + x / y - x + y;
    printf("Unused calculation: %f\n", result);
}

int ancient_algorithm(int n) {
    if (n <= 1) return 1;
    return ancient_algorithm(n - 1) + ancient_algorithm(n - 2);
}

void obsolete_printer(void) {
    printf("This is an obsolete printing function\n");
    for (int i = 0; i < 50; i++) {
        printf("Line %d\n", i);
    }
}

double forgotten_math_function(double a, double b, double c) {
    return (a * a + b * b + c * c) / (a + b + c);
}

void unused_string_processor(char* str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = temp;
    }
}

int legacy_validator(int* data, int size) {
    int valid_count = 0;
    for (int i = 0; i < size; i++) {
        if (data[i] % 2 == 0) {
            valid_count++;
        }
    }
    return valid_count;
}

void deprecated_initializer(void) {
    for (int i = 0; i < 1024; i++) {
        unused_cache[i] = i * 3;
    }
    for (int i = 0; i < 500; i++) {
        unused_coefficients[i] = i * 0.5;
    }
}

double old_computation_engine(int iterations) {
    double result = 1.0;
    for (int i = 0; i < iterations; i++) {
        result *= 1.001;
        result += 0.1 * i;
    }
    return result;
}

void unused_array_manipulator(int* arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int forgotten_search_function(int* array, int size, int target) {
    for (int i = 0; i < size; i++) {
        if (array[i] == target) {
            return i;
        }
    }
    return -1;
}

void legacy_data_formatter(char* output, int* data, int size) {
    int offset = 0;
    for (int i = 0; i < size; i++) {
        offset += sprintf(output + offset, "%d,", data[i]);
    }
}

// Main function - uses only core functionality
int main(int argc, char* argv[]) {
    printf("Starting legacy application...\n");
    
    // Initialize used data structures
    initialize_lookup_table();
    print_status(used_messages[0]);
    
    // Create test data
    int test_data[10] = {5, 12, 8, 23, 15, 7, 19, 11, 3, 9};
    
    // Validate inputs
    int all_valid = 1;
    for (int i = 0; i < 10; i++) {
        if (!validate_input(test_data[i])) {
            all_valid = 0;
            break;
        }
    }
    
    if (!all_valid) {
        printf("Input validation failed\n");
        return 1;
    }
    print_status(used_messages[2]);
    
    // Process the data
    print_status(used_messages[1]);
    process_data(test_data, 10);
    
    // Calculate statistics
    int sum = calculate_sum(test_data, 10);
    int avg = compute_average(test_data, 10);
    int max = find_maximum(test_data, 10);
    
    printf("Results:\n");
    printf("  Sum: %d\n", sum);
    printf("  Average: %d\n", avg);
    printf("  Maximum: %d\n", max);
    
    // Verify lookup table was initialized
    if (used_lookup_table[10] == 100) {
        print_status("Lookup table verified");
    }
    
    print_status(used_messages[3]);
    printf("Application completed successfully\n");
    
    return 0;
}