#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 0.0001

// External function declarations from matrix.c
extern void matrix_multiply(double* A, double* B, double* C, int m, int n, int p);
extern void matrix_add(double* A, double* B, double* C, int rows, int cols);
extern void matrix_transpose(double* A, double* AT, int rows, int cols);

// External function declarations from vector.c
extern double dot_product(double* a, double* b, int n);
extern double vector_norm(double* v, int n);
extern void vector_add(double* a, double* b, double* c, int n);
extern void vector_scale(double* v, double scalar, double* result, int n);

int test_matrix_multiply() {
    printf("Running test_matrix_multiply...\n");
    
    // 2x3 matrix A
    double A[] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};
    
    // 3x2 matrix B
    double B[] = {7.0, 8.0,
                  9.0, 10.0,
                  11.0, 12.0};
    
    // Result should be 2x2 matrix C
    double C[4] = {0};
    
    // Expected result: [58, 64; 139, 154]
    double expected[] = {58.0, 64.0, 139.0, 154.0};
    
    matrix_multiply(A, B, C, 2, 3, 2);
    
    for (int i = 0; i < 4; i++) {
        if (fabs(C[i] - expected[i]) > EPSILON) {
            printf("FAILED: matrix_multiply - expected %.2f, got %.2f at index %d\n", 
                   expected[i], C[i], i);
            return 1;
        }
    }
    
    printf("PASSED: test_matrix_multiply\n");
    return 0;
}

int test_matrix_add() {
    printf("Running test_matrix_add...\n");
    
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double B[] = {5.0, 6.0, 7.0, 8.0};
    double C[4] = {0};
    double expected[] = {6.0, 8.0, 10.0, 12.0};
    
    matrix_add(A, B, C, 2, 2);
    
    for (int i = 0; i < 4; i++) {
        if (fabs(C[i] - expected[i]) > EPSILON) {
            printf("FAILED: matrix_add - expected %.2f, got %.2f\n", expected[i], C[i]);
            return 1;
        }
    }
    
    printf("PASSED: test_matrix_add\n");
    return 0;
}

int test_matrix_transpose() {
    printf("Running test_matrix_transpose...\n");
    
    double A[] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0};
    double AT[6] = {0};
    double expected[] = {1.0, 4.0,
                        2.0, 5.0,
                        3.0, 6.0};
    
    matrix_transpose(A, AT, 2, 3);
    
    for (int i = 0; i < 6; i++) {
        if (fabs(AT[i] - expected[i]) > EPSILON) {
            printf("FAILED: matrix_transpose\n");
            return 1;
        }
    }
    
    printf("PASSED: test_matrix_transpose\n");
    return 0;
}

int test_dot_product() {
    printf("Running test_dot_product...\n");
    
    double a[] = {1.0, 2.0, 3.0, 4.0};
    double b[] = {5.0, 6.0, 7.0, 8.0};
    
    double result = dot_product(a, b, 4);
    double expected = 70.0; // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    
    if (fabs(result - expected) > EPSILON) {
        printf("FAILED: dot_product - expected %.2f, got %.2f\n", expected, result);
        return 1;
    }
    
    printf("PASSED: test_dot_product\n");
    return 0;
}

int test_vector_norm() {
    printf("Running test_vector_norm...\n");
    
    double v[] = {3.0, 4.0};
    double result = vector_norm(v, 2);
    double expected = 5.0; // sqrt(9 + 16) = sqrt(25) = 5
    
    if (fabs(result - expected) > EPSILON) {
        printf("FAILED: vector_norm - expected %.2f, got %.2f\n", expected, result);
        return 1;
    }
    
    printf("PASSED: test_vector_norm\n");
    return 0;
}

int test_vector_add() {
    printf("Running test_vector_add...\n");
    
    double a[] = {1.0, 2.0, 3.0};
    double b[] = {4.0, 5.0, 6.0};
    double c[3] = {0};
    double expected[] = {5.0, 7.0, 9.0};
    
    vector_add(a, b, c, 3);
    
    for (int i = 0; i < 3; i++) {
        if (fabs(c[i] - expected[i]) > EPSILON) {
            printf("FAILED: vector_add\n");
            return 1;
        }
    }
    
    printf("PASSED: test_vector_add\n");
    return 0;
}

int test_vector_scale() {
    printf("Running test_vector_scale...\n");
    
    double v[] = {2.0, 4.0, 6.0};
    double result[3] = {0};
    double scalar = 3.0;
    double expected[] = {6.0, 12.0, 18.0};
    
    vector_scale(v, scalar, result, 3);
    
    for (int i = 0; i < 3; i++) {
        if (fabs(result[i] - expected[i]) > EPSILON) {
            printf("FAILED: vector_scale\n");
            return 1;
        }
    }
    
    printf("PASSED: test_vector_scale\n");
    return 0;
}

int main() {
    int failed = 0;
    
    printf("=== Math Library Test Suite ===\n\n");
    
    failed += test_matrix_multiply();
    failed += test_matrix_add();
    failed += test_matrix_transpose();
    failed += test_dot_product();
    failed += test_vector_norm();
    failed += test_vector_add();
    failed += test_vector_scale();
    
    printf("\n=== Test Summary ===\n");
    if (failed == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d test(s) failed.\n", failed);
        return 1;
    }
}