/* Calculator Program - Main Entry Point */
/* This program demonstrates basic calculator operations */

#include <stdio.h>
#include <stdlib.h>
#include "operations.h"

int main() {
    /* Variable declarations - some intentionally problematic */
    int a = 10;
    int b = 5;
    int unused_var;  /* Unused variable */
    float temp;      /* Another unused variable */
    int result;      /* Uninitialized variable */
    float division_result;
    int multiplication;
    char status = 's';
    
    /* Print uninitialized variable - problematic */
    printf("Starting calculator with result: %d\n", result);
    
    /* Basic arithmetic operations */
    printf("Addition: %d + %d = %d\n", a, b, add(a, b));
    
    /* Multiplication operation */
    multiplication = multiply(4, 7);
    printf("Multiplication: 4 * 7 = %d\n", multiplication);
    
    /* Division operation with type mismatch in printf */
    division_result = divide(20.0, 4.0);
    printf("Division result: %d\n", division_result);  /* Format mismatch - should be %f */
    
    /* Calling an undeclared function - implicit declaration */
    int power_result = power(2, 3);
    printf("Power: 2^3 = %d\n", power_result);
    
    /* Subtraction with wrong format specifier */
    float subtraction_result = 15.5;
    printf("Subtraction result: %d\n", subtraction_result);  /* Type mismatch */
    
    /* Another implicit function call */
    modulo_operation(10, 3);
    
    /* Using wrong type in comparison */
    if (status) {
        printf("Status is active\n");
    }
    
    /* Missing format argument */
    printf("Final calculation: %d and %d\n", a);
    
    /* Signed/unsigned comparison issue */
    unsigned int positive = 5;
    if (b > positive) {
        printf("Comparison done\n");
    }
    
    printf("Calculator operations completed\n");
    
    return 0;
}