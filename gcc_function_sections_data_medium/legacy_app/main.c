#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

int main(void) {
    printf("Legacy Application Starting...\n");
    printf("=====================================\n");
    
    // Test basic arithmetic utility
    int sum_result = calculate_sum(10, 20);
    printf("Sum calculation: 10 + 20 = %d\n", sum_result);
    
    // Test string formatting utility
    char* formatted = format_string("test");
    if (formatted != NULL) {
        printf("Formatted string: %s\n", formatted);
        free(formatted);
    }
    
    // Test validation utility
    int validation_result = validate_input(42);
    printf("Input validation (42): %s\n", 
           validation_result ? "PASSED" : "FAILED");
    
    printf("=====================================\n");
    printf("Application completed successfully\n");
    
    return 0;
}