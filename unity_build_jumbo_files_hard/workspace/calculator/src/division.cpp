#include "../include/division.h"

double divide(int a, int b) {
    if (b == 0) {
        return 0.0;
    }
    return static_cast<double>(a) / static_cast<double>(b);
}