#include "mathutils.h"

int math_add(int a, int b) {
    return a + b;
}

int math_multiply(int a, int b) {
    return a * b;
}

int math_subtract(int a, int b) {
    return a - b;
}

int math_factorial(int n) {
    if (n < 0) {
        return -1;
    }
    if (n == 0 || n == 1) {
        return 1;
    }
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

int math_power(int base, int exponent) {
    if (exponent < 0) {
        return 0;
    }
    if (exponent == 0) {
        return 1;
    }
    int result = 1;
    for (int i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}