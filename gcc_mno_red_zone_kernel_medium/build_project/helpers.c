#include <stdio.h>
#include <stdlib.h>

/*
 * Helper functions for the build project
 */

// Check if a number is even
int is_even(int n) {
    return n % 2 == 0;
}

// Double the value of an integer
int double_value(int n) {
    return n * 2;
}

// Print an integer value
void print_int(int n) {
    printf("Value: %d\n", n);
}

// Calculate the sum of two integers
int add_numbers(int a, int b) {
    return a + b;
}

// Check if a number is positive
int is_positive(int n) {
    return n > 0;
}