#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix_ops.h"

void init_matrix(double* matrix, int size) {
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

void print_matrix_element(double* matrix, int row, int col, int size) {
    if (row < 0 || row >= size || col < 0 || col >= size) {
        printf("Error: Index out of bounds\n");
        return;
    }
    printf("matrix[%d][%d] = %f\n", row, col, matrix[row * size + col]);
}

int validate_matrix(double* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        if (isnan(matrix[i]) || isinf(matrix[i])) {
            return 0;
        }
    }
    return 1;
}

double get_matrix_element(double* matrix, int row, int col, int size) {
    return matrix[row * size + col];
}

void set_matrix_element(double* matrix, int row, int col, int size, double value) {
    matrix[row * size + col] = value;
}

double sum_matrix_elements(double* matrix, int size) {
    double sum = 0.0;
    for (int i = 0; i < size * size; i++) {
        sum += matrix[i];
    }
    return sum;
}

void zero_matrix(double* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = 0.0;
    }
}

void copy_matrix(double* dest, double* src, int size) {
    for (int i = 0; i < size * size; i++) {
        dest[i] = src[i];
    }
}

double max_matrix_element(double* matrix, int size) {
    double max_val = matrix[0];
    for (int i = 1; i < size * size; i++) {
        if (matrix[i] > max_val) {
            max_val = matrix[i];
        }
    }
    return max_val;
}

double min_matrix_element(double* matrix, int size) {
    double min_val = matrix[0];
    for (int i = 1; i < size * size; i++) {
        if (matrix[i] < min_val) {
            min_val = matrix[i];
        }
    }
    return min_val;
}