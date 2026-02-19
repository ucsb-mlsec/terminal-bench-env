#include <stdio.h>
#include "mathops.h"

int main() {
    int result1, result2;
    
    // Test add_numbers function
    result1 = add_numbers(5, 3);
    if (result1 != 8) {
        printf("ERROR: add_numbers(5, 3) returned %d, expected 8\n", result1);
        return 1;
    }
    
    // Test multiply_numbers function
    result2 = multiply_numbers(4, 7);
    if (result2 != 28) {
        printf("ERROR: multiply_numbers(4, 7) returned %d, expected 28\n", result2);
        return 1;
    }
    
    // Both tests passed
    printf("SUCCESS\n");
    return 0;
}