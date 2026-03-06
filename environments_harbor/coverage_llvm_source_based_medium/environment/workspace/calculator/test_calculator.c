/*
 * test_calculator.c
 * Test program for calculator library
 * Note: This test suite has incomplete coverage - not all functions are fully tested
 */

#include <stdio.h>

// External function declarations from calculator library
int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b);
int divide(int a, int b);
int modulo(int a, int b);

int main() {
    int result;
    
    printf("Calculator Test Suite\n");
    printf("=====================\n\n");
    
    // Test addition thoroughly
    result = add(5, 3);
    printf("add(5, 3) = %d\n", result);
    
    result = add(10, 20);
    printf("add(10, 20) = %d\n", result);
    
    result = add(-5, 5);
    printf("add(-5, 5) = %d\n", result);
    
    // Test subtraction
    result = subtract(10, 3);
    printf("subtract(10, 3) = %d\n", result);
    
    result = subtract(5, 10);
    printf("subtract(5, 10) = %d\n", result);
    
    // Test multiplication partially
    result = multiply(4, 5);
    printf("multiply(4, 5) = %d\n", result);
    
    // Note: divide() and modulo() are NOT tested
    // This creates incomplete test coverage
    
    printf("\nTests completed\n");
    
    return 0;
}