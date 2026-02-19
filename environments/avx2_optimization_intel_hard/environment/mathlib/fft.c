#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "mathlib.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declaration
static void fft_recursive(double complex *input, double complex *output, int n, int stride);

/**
 * Compute FFT using Cooley-Tukey algorithm
 * 
 * @param input Input array of complex numbers
 * @param output Output array for FFT result
 * @param n Size of the input array (must be power of 2)
 * @return 0 on success, -1 on error
 */
int fft_compute(double complex *input, double complex *output, int n) {
    if (n <= 0 || (n & (n - 1)) != 0) {
        fprintf(stderr, "Error: FFT size must be a power of 2\n");
        return -1;
    }
    
    if (input == NULL || output == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to fft_compute\n");
        return -1;
    }
    
    fft_recursive(input, output, n, 1);
    return 0;
}

/**
 * Recursive FFT implementation using divide-and-conquer
 * 
 * @param input Input array of complex numbers
 * @param output Output array for FFT result
 * @param n Number of elements to process
 * @param stride Stride for accessing elements
 */
static void fft_recursive(double complex *input, double complex *output, int n, int stride) {
    if (n == 1) {
        output[0] = input[0];
        return;
    }
    
    int half_n = n / 2;
    
    // Allocate temporary arrays for even and odd elements
    double complex *even = (double complex *)malloc(half_n * sizeof(double complex));
    double complex *odd = (double complex *)malloc(half_n * sizeof(double complex));
    
    if (even == NULL || odd == NULL) {
        fprintf(stderr, "Error: Memory allocation failed in fft_recursive\n");
        free(even);
        free(odd);
        return;
    }
    
    // Recursively compute FFT of even and odd indexed elements
    fft_recursive(input, even, half_n, stride * 2);
    fft_recursive(input + stride, odd, half_n, stride * 2);
    
    // Combine results using butterfly operations
    for (int k = 0; k < half_n; k++) {
        double angle = -2.0 * M_PI * k / n;
        double complex twiddle = cexp(I * angle);
        double complex t = twiddle * odd[k];
        
        output[k] = even[k] + t;
        output[k + half_n] = even[k] - t;
    }
    
    free(even);
    free(odd);
}

/**
 * Compute inverse FFT
 * 
 * @param input Input array of complex numbers
 * @param output Output array for IFFT result
 * @param n Size of the input array (must be power of 2)
 * @return 0 on success, -1 on error
 */
int ifft_compute(double complex *input, double complex *output, int n) {
    if (n <= 0 || (n & (n - 1)) != 0) {
        fprintf(stderr, "Error: IFFT size must be a power of 2\n");
        return -1;
    }
    
    // Take complex conjugate of input
    double complex *conjugated = (double complex *)malloc(n * sizeof(double complex));
    if (conjugated == NULL) {
        fprintf(stderr, "Error: Memory allocation failed in ifft_compute\n");
        return -1;
    }
    
    for (int i = 0; i < n; i++) {
        conjugated[i] = conj(input[i]);
    }
    
    // Compute FFT of conjugated input
    int result = fft_compute(conjugated, output, n);
    
    if (result == 0) {
        // Take complex conjugate and scale by 1/n
        for (int i = 0; i < n; i++) {
            output[i] = conj(output[i]) / n;
        }
    }
    
    free(conjugated);
    return result;
}