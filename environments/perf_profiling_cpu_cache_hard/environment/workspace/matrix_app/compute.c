#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 512

// Matrix stored in row-major order
double matrix_a[SIZE][SIZE];
double matrix_b[SIZE][SIZE];
double matrix_result[SIZE][SIZE];

void initialize_matrices() {
    // Deterministic initialization
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix_a[i][j] = (double)(i * SIZE + j) / 1000.0;
            matrix_b[i][j] = (double)(j * SIZE + i) / 1000.0;
            matrix_result[i][j] = 0.0;
        }
    }
}

// CACHE-INEFFICIENT matrix multiplication
// Accessing matrices in column-major order (j, then i) while they are stored row-major
// This causes excessive cache misses
void inefficient_matrix_multiply() {
    // Intentionally inefficient: iterating columns first on row-major storage
    for (int j = 0; j < SIZE; j++) {           // column index outer loop
        for (int i = 0; i < SIZE; i++) {       // row index middle loop
            for (int k = 0; k < SIZE; k++) {   // inner loop
                // Accessing matrix_a[i][k] and matrix_b[k][j] with poor locality
                matrix_result[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
}

double compute_checksum() {
    double sum = 0.0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            sum += matrix_result[i][j];
        }
    }
    return sum;
}

int main() {
    initialize_matrices();
    inefficient_matrix_multiply();
    double checksum = compute_checksum();
    printf("Checksum: %.2f\n", checksum);
    return 0;
}