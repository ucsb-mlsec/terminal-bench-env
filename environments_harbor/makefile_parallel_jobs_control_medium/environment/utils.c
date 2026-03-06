#include <stdio.h>
#include "math_ops.h"
#include "utils.h"

int square(int x) {
    return multiply(x, x);
}

int increment(int x) {
    return add(x, 1);
}

int decrement(int x) {
    return subtract(x, 1);
}

int cube(int x) {
    int squared = multiply(x, x);
    return multiply(squared, x);
}

void print_calculation(int a, int b) {
    printf("Sum: %d\n", add(a, b));
    printf("Product: %d\n", multiply(a, b));
    printf("Difference: %d\n", subtract(a, b));
}

int double_value(int x) {
    return add(x, x);
}