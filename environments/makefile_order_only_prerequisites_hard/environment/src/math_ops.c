#include <stdio.h>

int multiply(int a, int b) {
    int result;
    result = a * b;
    printf("Multiplying %d and %d\n", a, b);
    return result;
}

double divide(double a, double b) {
    double result;
    if (b == 0.0) {
        printf("Error: Division by zero\n");
        return 0.0;
    }
    result = a / b;
    printf("Dividing %.2f by %.2f\n", a, b);
    return result;
}