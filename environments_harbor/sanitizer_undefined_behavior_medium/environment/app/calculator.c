#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to perform array-based calculations
int calculate_sum(int *arr, int size) {
    printf("Running test 1: Array sum calculation...\n");
    int sum = 0;
    // UB: Buffer overflow - accessing beyond array bounds
    for (int i = 0; i <= size; i++) {
        sum += arr[i];
    }
    return sum;
}

// Function to calculate factorial with overflow
int factorial(int n) {
    printf("Running test 2: Factorial calculation...\n");
    int result = 1;
    // UB: Signed integer overflow
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Function to perform division operations
int divide_numbers(int a, int b) {
    printf("Running test 3: Division operation...\n");
    // UB: Division by zero
    return a / b;
}

// Function to perform bitwise operations
int shift_operation(int value, int shift_amount) {
    printf("Running test 4: Bitwise shift operation...\n");
    // UB: Invalid shift amount (shifting by amount >= width of type)
    return value << shift_amount;
}

// Function to work with pointers
int process_data(int *data) {
    printf("Running test 5: Pointer operation...\n");
    // UB: Null pointer dereference
    return *data + 10;
}

// Function using uninitialized memory
int compute_value() {
    printf("Running test 6: Value computation...\n");
    int x;
    // UB: Use of uninitialized variable
    return x * 2;
}

// Function with memory management
void memory_operation() {
    printf("Running test 7: Memory operation...\n");
    int *ptr = (int *)malloc(sizeof(int));
    *ptr = 42;
    free(ptr);
    // UB: Use after free
    printf("Value: %d\n", *ptr);
}

int main() {
    printf("Starting calculator application...\n\n");
    
    // Test 1: Array overflow
    int numbers[5] = {1, 2, 3, 4, 5};
    int sum = calculate_sum(numbers, 5);
    printf("Sum result: %d\n\n", sum);
    
    // Test 2: Integer overflow
    int fact = factorial(50);
    printf("Factorial result: %d\n\n", fact);
    
    // Test 3: Division by zero
    int div_result = divide_numbers(100, 0);
    printf("Division result: %d\n\n", div_result);
    
    // Test 4: Invalid shift
    int shift_result = shift_operation(1, 35);
    printf("Shift result: %d\n\n", shift_result);
    
    // Test 5: Null pointer dereference
    int *null_ptr = NULL;
    int ptr_result = process_data(null_ptr);
    printf("Pointer result: %d\n\n", ptr_result);
    
    // Test 6: Uninitialized variable
    int uninit_result = compute_value();
    printf("Uninitialized result: %d\n\n", uninit_result);
    
    // Test 7: Use after free
    memory_operation();
    
    printf("\nCalculator tests completed.\n");
    return 0;
}