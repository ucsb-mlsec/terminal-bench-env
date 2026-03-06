#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SIZE 100
#define BUFFER_SIZE 50

// Function to calculate the sum of array elements
int calculate_sum(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i <= size; i++) {  // Potential off-by-one error
        sum += arr[i];
    }
    return sum;
}

// Function to copy data between buffers
void copy_buffer(char *dest, const char *src, int length) {
    for (int i = 0; i < length; i++) {
        dest[i] = src[i];  // Potential buffer overflow
    }
}

// Function to find maximum element in array
int find_max(int *data, int count) {
    int max = data[0];  // Potential null pointer dereference
    for (int i = 1; i < count; i++) {
        if (data[i] > max) {
            max = data[i];
        }
    }
    return max;
}

// Function to allocate and initialize array
int* create_array(int size) {
    int *arr = (int*)malloc(size * sizeof(int));
    if (arr != NULL) {
        for (int i = 0; i < size; i++) {
            arr[i] = i * 2;
        }
    }
    return arr;  // Caller responsible for freeing
}

// Function to process string input
void process_string(char *input) {
    char buffer[BUFFER_SIZE];
    strcpy(buffer, input);  // Potential buffer overflow
    printf("Processed: %s\n", buffer);
}

// Function to swap two integers using pointers
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Function with division operation
int safe_divide(int numerator, int denominator) {
    return numerator / denominator;  // Potential division by zero
}

int main(int argc, char *argv[]) {
    int numbers[MAX_SIZE];
    int size = 10;
    
    // Initialize array
    for (int i = 0; i < size; i++) {
        numbers[i] = i + 1;
    }
    
    // Calculate sum with potential bounds issue
    int total = calculate_sum(numbers, size);
    printf("Sum: %d\n", total);
    
    // Find maximum element
    int max_val = find_max(numbers, size);
    printf("Maximum: %d\n", max_val);
    
    // Test swap function
    int x = 10, y = 20;
    swap(&x, &y);
    printf("After swap: x=%d, y=%d\n", x, y);
    
    // Dynamic allocation test
    int *dynamic_arr = create_array(15);
    if (dynamic_arr != NULL) {
        int dyn_max = find_max(dynamic_arr, 15);
        printf("Dynamic array max: %d\n", dyn_max);
        // Missing free - memory leak
    }
    
    // String processing with potential overflow
    char test_input[100];
    strcpy(test_input, "Test string for verification");
    process_string(test_input);
    
    // Division operation
    int result = safe_divide(100, 5);
    printf("Division result: %d\n", result);
    
    // Potential null pointer access
    int *ptr = NULL;
    if (argc > 5) {
        *ptr = 42;  // Null pointer dereference
    }
    
    return 0;
}