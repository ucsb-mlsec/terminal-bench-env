#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern double compute(double);

int main() {
    double input = 10.0;
    double result = compute(input);
    
    printf("Modern app: compute(%.2f) = %.2f\n", input, result);
    
    // Version 2.0 should return floating-point result (e.g., input * 1.5 = 15.0)
    double expected = 15.0;
    
    if (fabs(result - expected) < 0.01) {
        printf("PASS: Got expected v2.0 floating-point result\n");
        return 0;
    } else {
        printf("FAIL: Expected %.2f but got %.2f\n", expected, result);
        return 1;
    }
}