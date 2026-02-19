#include <stdio.h>
#include "calc.h"

int add_numbers(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int compute_value(int x) {
    int result = x * x + x;
    return result;
}

int subtract_numbers(int a, int b) {
    return a - b;
}

int divide_numbers(int a, int b) {
    if (b == 0) {
        printf("Error: Division by zero\n");
        return 0;
    }
    return a / b;
}

int square(int x) {
    return x * x;
}

int cube(int x) {
    return x * x * x;
}