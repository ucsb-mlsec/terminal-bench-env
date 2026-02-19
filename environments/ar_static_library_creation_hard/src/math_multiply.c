#include <stdio.h>

int multiply(int a, int b) {
    return a * b;
}

int math_multiply(int a, int b) {
    printf("Math library multiply: %d * %d = %d\n", a, b, a * b);
    return a * b;
}