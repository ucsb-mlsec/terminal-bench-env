#include "math.h"

int power(int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; i++) {
        result *= base;
    }
    return result;
}

double power_double(double base, double exp) {
    double result = 1.0;
    int e = (int)exp;
    for (int i = 0; i < e; i++) {
        result *= base;
    }
    return result;
}