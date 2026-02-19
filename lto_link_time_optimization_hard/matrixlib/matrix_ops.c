#include <stdio.h>
#include <stdlib.h>
#include "matrix_ops.h"
#include "matrix_helpers.h"

void matrix_multiply(double* A, double* B, double* C, int size) {
    int i, j, k;
    
    initialize_matrix(C, size);
    
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            double sum = 0.0;
            for (k = 0; k < size; k++) {
                sum += get_element(A, i, k, size) * get_element(B, k, j, size);
            }
            set_element(C, i, j, size, sum);
        }
    }
    
    validate_matrix(C, size);
}

void matrix_add(double* A, double* B, double* C, int size) {
    int i, j;
    
    initialize_matrix(C, size);
    
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            double a_val = get_element(A, i, j, size);
            double b_val = get_element(B, i, j, size);
            double result = compute_sum(a_val, b_val);
            set_element(C, i, j, size, result);
        }
    }
    
    validate_matrix(C, size);
}

void matrix_transpose(double* A, double* B, int size) {
    int i, j;
    
    initialize_matrix(B, size);
    
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            double val = get_element(A, i, j, size);
            set_element(B, j, i, size, val);
        }
    }
    
    validate_matrix(B, size);
}

void matrix_scale(double* A, double* B, double scalar, int size) {
    int i, j;
    
    initialize_matrix(B, size);
    
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            double val = get_element(A, i, j, size);
            double result = compute_product(val, scalar);
            set_element(B, i, j, size, result);
        }
    }
    
    validate_matrix(B, size);
}