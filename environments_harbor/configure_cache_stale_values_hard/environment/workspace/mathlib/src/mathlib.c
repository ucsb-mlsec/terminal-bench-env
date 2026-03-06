#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* MathLib - A simple mathematical library */

double calculate_distance(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx*dx + dy*dy);
}

double calculate_area(double radius) {
    return 3.14159265358979323846 * radius * radius;
}

double calculate_circumference(double radius) {
    return 2.0 * 3.14159265358979323846 * radius;
}

double factorial(int n) {
    if (n <= 1) return 1.0;
    return n * factorial(n - 1);
}

int main() {
    printf("MathLib Test Suite\n");
    printf("==================\n");
    printf("Distance from (0,0) to (3,4): %f\n", calculate_distance(0, 0, 3, 4));
    printf("Area of circle with radius 5: %f\n", calculate_area(5.0));
    printf("Circumference with radius 5: %f\n", calculate_circumference(5.0));
    printf("Factorial of 5: %f\n", factorial(5));
    return 0;
}