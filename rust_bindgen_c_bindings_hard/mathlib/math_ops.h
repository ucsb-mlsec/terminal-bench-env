#ifndef MATH_OPS_H
#define MATH_OPS_H

/* Mathematical Operations Library
 * Provides basic mathematical functions and utilities
 */

/* Constants */
#define PI 3.14159265358979323846
#define E 2.71828182845904523536
#define MAX_VALUE 1000
#define MIN_VALUE -1000
#define EPSILON 0.0000001

/* Type definitions */

/* Point structure representing a 2D coordinate */
typedef struct {
    double x;
    double y;
} Point;

/* Vector structure for 2D vectors */
typedef struct {
    float dx;
    float dy;
} Vector2D;

/* Function pointer type for mathematical operations */
typedef int (*MathFunc)(int, int);

/* Complex number structure */
typedef struct {
    double real;
    double imag;
} Complex;

/* Function declarations */

/**
 * Adds two integers
 * @param a First integer
 * @param b Second integer
 * @return Sum of a and b
 */
int add(int a, int b);

/**
 * Subtracts two integers
 * @param a First integer
 * @param b Second integer
 * @return Difference of a and b
 */
int subtract(int a, int b);

/**
 * Multiplies two double precision numbers
 * @param x First number
 * @param y Second number
 * @return Product of x and y
 */
double multiply(double x, double y);

/**
 * Divides two floating point numbers
 * @param numerator The dividend
 * @param divisor The divisor
 * @return Result of division (numerator / divisor)
 */
float divide(float numerator, float divisor);

/**
 * Calculates power (base^exponent)
 * @param base The base number
 * @param exponent The exponent
 * @return base raised to the power of exponent
 */
int power(int base, int exponent);

/**
 * Calculates the factorial of a number
 * @param n The number
 * @return Factorial of n
 */
long long factorial(int n);

/**
 * Calculates the distance between two points
 * @param p1 First point
 * @param p2 Second point
 * @return Euclidean distance between p1 and p2
 */
double distance(Point p1, Point p2);

/**
 * Calculates the magnitude of a 2D vector
 * @param v The vector
 * @return Magnitude of the vector
 */
float vector_magnitude(Vector2D v);

/**
 * Calculates the absolute value of an integer
 * @param x The integer
 * @return Absolute value of x
 */
int absolute(int x);

/**
 * Returns the maximum of two integers
 * @param a First integer
 * @param b Second integer
 * @return Maximum of a and b
 */
int max(int a, int b);

/**
 * Returns the minimum of two integers
 * @param a First integer
 * @param b Second integer
 * @return Minimum of a and b
 */
int min(int a, int b);

/**
 * Computes square root approximation using integer arithmetic
 * @param n The number
 * @return Approximate square root of n
 */
int int_sqrt(int n);

/**
 * Adds two complex numbers
 * @param a First complex number
 * @param b Second complex number
 * @return Sum of complex numbers
 */
Complex complex_add(Complex a, Complex b);

/**
 * Checks if a number is prime
 * @param n The number to check
 * @return 1 if prime, 0 otherwise
 */
int is_prime(int n);

#endif /* MATH_OPS_H */