#include <stdio.h>

// Version 1.0 implementation - returns integer
int compute_v1(int x) {
    printf("Using compute v1.0 (integer version)\n")
    return x * 2;
}

// Incorrect symbol versioning - missing @ symbol
__asm__(".symver compute_v1,computeLIBMATH_1.0");

// Version 2.0 implementation - returns double
double compute_v2(double x) {
    printf("Using compute v2.0 (floating-point version)\n");
    return x * 1.5;
}

// Incorrect default version symbol - should use @@ not @
__asm__(".symver compute_v2,compute@LIBMATH_2.0");

// Missing version node declarations that should be in version script
// These functions provide the actual implementations

int compute(int x) {
    return compute_v1(x);
}

double compute(double x) {
    return compute_v2(x);
}