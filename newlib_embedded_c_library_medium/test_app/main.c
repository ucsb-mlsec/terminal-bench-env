#include <stdio.h>
#include <string.h>
#include "../libutils/string_utils.h"
#include "../libutils/math_utils.h"
#include "../libutils/array_utils.h"

int main() {
    printf("=== Testing Utility Library ===\n\n");
    
    // Test 1: String utilities
    printf("Test 1: String Utilities\n");
    char test_string[] = "Hello World";
    printf("Original string: %s\n", test_string);
    str_reverse(test_string);
    printf("Reversed string: %s\n\n", test_string);
    
    // Test 2: Math utilities
    printf("Test 2: Math Utilities\n");
    int fact_result = factorial(5);
    printf("factorial(5) = %d\n", fact_result);
    
    int prime_check = is_prime(17);
    printf("is_prime(17) = %d (1=prime, 0=not prime)\n\n", prime_check);
    
    // Test 3: Array utilities
    printf("Test 3: Array Utilities\n");
    int test_array[] = {15, 42, 8, 23, 56, 11};
    int array_size = 6;
    
    printf("Array: ");
    for (int i = 0; i < array_size; i++) {
        printf("%d ", test_array[i]);
    }
    printf("\n");
    
    int max_value = array_max(test_array, array_size);
    printf("array_max() = %d\n", max_value);
    
    int sum_value = array_sum(test_array, array_size);
    printf("array_sum() = %d\n\n", sum_value);
    
    printf("=== All Tests Complete ===\n");
    
    return 0;
}