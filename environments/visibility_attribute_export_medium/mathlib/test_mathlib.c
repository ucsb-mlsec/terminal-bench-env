#include <stdio.h>
#include "mathlib.h"

int main() {
    int all_passed = 1;
    
    // Test 1: add_numbers function
    // Expected: 5.0 + 3.0 = 8.0
    double add_result = add_numbers(5.0, 3.0);
    if (add_result == 8.0) {
        printf("Test 1 PASSED: add_numbers(5.0, 3.0) = %.1f\n", add_result);
    } else {
        printf("Test 1 FAILED: add_numbers(5.0, 3.0) = %.1f (expected 8.0)\n", add_result);
        all_passed = 0;
    }
    
    // Test 2: multiply_numbers function
    // Expected: 4.0 * 2.5 = 10.0
    double mult_result = multiply_numbers(4.0, 2.5);
    if (mult_result == 10.0) {
        printf("Test 2 PASSED: multiply_numbers(4.0, 2.5) = %.1f\n", mult_result);
    } else {
        printf("Test 2 FAILED: multiply_numbers(4.0, 2.5) = %.1f (expected 10.0)\n", mult_result);
        all_passed = 0;
    }
    
    // Test 3: calculate_average function
    // Expected: average of {10.0, 20.0, 30.0, 40.0, 50.0} = 30.0
    double numbers[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    double avg_result = calculate_average(numbers, 5);
    if (avg_result == 30.0) {
        printf("Test 3 PASSED: calculate_average([10,20,30,40,50], 5) = %.1f\n", avg_result);
    } else {
        printf("Test 3 FAILED: calculate_average([10,20,30,40,50], 5) = %.1f (expected 30.0)\n", avg_result);
        all_passed = 0;
    }
    
    // Print overall result
    if (all_passed) {
        printf("\nAll tests PASSED!\n");
        return 0;
    } else {
        printf("\nSome tests FAILED!\n");
        return 1;
    }
}