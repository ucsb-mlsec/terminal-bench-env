#include <stdio.h>
#include <stdlib.h>
#include "tools.h"

int main() {
    printf("=== Testing Custom Tools Library ===\n\n");
    
    // Test 1: Process some data
    printf("Test 1: Processing data...\n");
    int data[] = {5, 10, 15, 20, 25};
    int count = 5;
    process_data(data, count);
    printf("\n");
    
    // Test 2: Calculate result with two numbers
    printf("Test 2: Calculating result...\n");
    int a = 10;
    int b = 20;
    int result = calculate_result(a, b);
    printf("Result of calculate_result(%d, %d) = %d\n", a, b, result);
    printf("\n");
    
    // Test 3: Format output with a value
    printf("Test 3: Formatting output...\n");
    int value = 42;
    format_output(value);
    printf("\n");
    
    // Test 4: Additional calculation
    printf("Test 4: Another calculation...\n");
    int x = 100;
    int y = 50;
    int sum = calculate_result(x, y);
    printf("Result of calculate_result(%d, %d) = %d\n", x, y, sum);
    printf("\n");
    
    // Test 5: Process different data set
    printf("Test 5: Processing another dataset...\n");
    int data2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    process_data(data2, 10);
    printf("\n");
    
    printf("=== All tests completed successfully! ===\n");
    
    return 0;
}