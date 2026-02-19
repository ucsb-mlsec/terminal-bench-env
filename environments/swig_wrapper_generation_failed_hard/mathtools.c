#include <stdio.h>

/* Simple mathematical functions for Python binding */

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

double square(double x) {
    return x * x;
}

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}