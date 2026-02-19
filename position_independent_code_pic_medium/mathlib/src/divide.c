#include "math.h"

int divide(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}

double divide_double(double a, double b) {
    if (b == 0.0) return 0.0;
    return a / b;
}