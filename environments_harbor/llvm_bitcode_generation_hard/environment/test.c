#include <stdio.h>
#include "mathlib.h"

int main() {
    printf("=== Mathematical Library Test Suite ===\n\n");
    
    // Test 1: vector_add
    printf("Test 1: Vector Addition\n");
    double v1[] = {1.0, 2.0, 3.0, 4.0};
    double v2[] = {5.0, 6.0, 7.0, 8.0};
    double v_result[4];
    vector_add(v1, v2, v_result, 4);
    printf("Vector 1: [1.0, 2.0, 3.0, 4.0]\n");
    printf("Vector 2: [5.0, 6.0, 7.0, 8.0]\n");
    printf("Result: [%.1f, %.1f, %.1f, %.1f]\n\n", 
           v_result[0], v_result[1], v_result[2], v_result[3]);
    
    // Test 2: vector_dot
    printf("Test 2: Vector Dot Product\n");
    double v3[] = {2.0, 3.0, 4.0};
    double v4[] = {1.0, 2.0, 3.0};
    double dot_result = vector_dot(v3, v4, 3);
    printf("Vector 1: [2.0, 3.0, 4.0]\n");
    printf("Vector 2: [1.0, 2.0, 3.0]\n");
    printf("Dot Product: %.1f\n\n", dot_result);
    
    // Test 3: vector_magnitude
    printf("Test 3: Vector Magnitude\n");
    double v5[] = {3.0, 4.0};
    double magnitude = vector_magnitude(v5, 2);
    printf("Vector: [3.0, 4.0]\n");
    printf("Magnitude: %.1f\n\n", magnitude);
    
    // Test 4: matrix_multiply
    printf("Test 4: Matrix Multiplication\n");
    double m1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};  // 2x3 matrix
    double m2[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};  // 3x2 matrix
    double m_result[4];  // 2x2 result
    matrix_multiply(m1, m2, m_result, 2, 3, 2);
    printf("Matrix A (2x3): [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\n");
    printf("Matrix B (3x2): [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]\n");
    printf("Result (2x2): [[%.1f, %.1f], [%.1f, %.1f]]\n\n",
           m_result[0], m_result[1], m_result[2], m_result[3]);
    
    // Test 5: matrix_transpose
    printf("Test 5: Matrix Transpose\n");
    double m3[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};  // 2x3 matrix
    double m_trans[6];  // 3x2 result
    matrix_transpose(m3, m_trans, 2, 3);
    printf("Original (2x3): [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\n");
    printf("Transposed (3x2): [[%.1f, %.1f], [%.1f, %.1f], [%.1f, %.1f]]\n\n",
           m_trans[0], m_trans[1], m_trans[2], m_trans[3], m_trans[4], m_trans[5]);
    
    // Test 6: matrix_trace
    printf("Test 6: Matrix Trace\n");
    double m4[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};  // 3x3 matrix
    double trace = matrix_trace(m4, 3);
    printf("Matrix (3x3): [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]\n");
    printf("Trace: %.1f\n\n", trace);
    
    // Test 7: compute_mean
    printf("Test 7: Compute Mean\n");
    double data1[] = {2.0, 4.0, 6.0, 8.0, 10.0};
    double mean = compute_mean(data1, 5);
    printf("Data: [2.0, 4.0, 6.0, 8.0, 10.0]\n");
    printf("Mean: %.1f\n\n", mean);
    
    // Test 8: compute_variance
    printf("Test 8: Compute Variance\n");
    double data2[] = {2.0, 4.0, 6.0, 8.0, 10.0};
    double variance = compute_variance(data2, 5);
    printf("Data: [2.0, 4.0, 6.0, 8.0, 10.0]\n");
    printf("Variance: %.1f\n\n", variance);
    
    // Test 9: compute_stddev
    printf("Test 9: Compute Standard Deviation\n");
    double data3[] = {2.0, 4.0, 6.0, 8.0, 10.0};
    double stddev = compute_stddev(data3, 5);
    printf("Data: [2.0, 4.0, 6.0, 8.0, 10.0]\n");
    printf("Standard Deviation: %.2f\n\n", stddev);
    
    printf("=== All Tests Completed Successfully ===\n");
    return 0;
}