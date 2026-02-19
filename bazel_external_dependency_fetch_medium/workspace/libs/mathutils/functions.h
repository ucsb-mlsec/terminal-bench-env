#ifndef MATHUTILS_FUNCTIONS_H
#define MATHUTILS_FUNCTIONS_H

namespace mathutils {

inline int add(int a, int b) {
    return a + b;
}

inline int subtract(int a, int b) {
    return a - b;
}

inline int multiply(int a, int b) {
    return a * b;
}

inline double divide(double a, double b) {
    return (b != 0.0) ? (a / b) : 0.0;
}

inline int square(int x) {
    return x * x;
}

} // namespace mathutils

#endif // MATHUTILS_FUNCTIONS_H