#include "math_ops.h"

int add(int a, int b) {
    return a + b;
}

double multiply(double x, double y) {
    return x * y;
}

double divide(double numerator, double divisor) {
    if (divisor == 0.0) {
        return 0.0;
    }
    return numerator / divisor;
}

double power(double base, int exponent) {
    double result = 1.0;
    int abs_exp = exponent < 0 ? -exponent : exponent;
    
    for (int i = 0; i < abs_exp; i++) {
        result *= base;
    }
    
    if (exponent < 0) {
        return 1.0 / result;
    }
    
    return result;
}

Point create_point(double x, double y) {
    Point p;
    p.x = x;
    p.y = y;
    return p;
}

double point_distance(Point p1, Point p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double dist_squared = dx * dx + dy * dy;
    
    // Simple square root approximation using Newton's method
    if (dist_squared == 0.0) {
        return 0.0;
    }
    
    double estimate = dist_squared / 2.0;
    for (int i = 0; i < 10; i++) {
        estimate = (estimate + dist_squared / estimate) / 2.0;
    }
    
    return estimate;
}