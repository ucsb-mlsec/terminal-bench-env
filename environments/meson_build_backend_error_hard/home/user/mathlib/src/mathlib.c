#include <stdio.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>

double add_numbers(double a, double b) {
    return a + b;
}

double multiply_numbers(double a, double b) {
    return a * b;
}

double compute_factorial(int n) {
    if (n < 0) {
        return -1.0;
    }
    return gsl_sf_gamma(n + 1);
}

double power_of_e(double x) {
    return gsl_sf_exp(x);
}

int is_finite_number(double x) {
    return gsl_finite(x);
}

void print_math_constants() {
    printf("Pi: %f\n", M_PI);
    printf("e: %f\n", M_E);
    printf("GSL Pi: %f\n", M_PI);
}