#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 512

double A[SIZE][SIZE];
double B[SIZE][SIZE];
double C[SIZE][SIZE];

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void initialize_matrices() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            A[i][j] = (double)(i + j);
            B[i][j] = (double)(i - j);
            C[i][j] = 0.0;
        }
    }
}

void matrix_multiply() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

double compute_checksum() {
    double sum = 0.0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            sum += C[i][j];
        }
    }
    return sum;
}

int main() {
    printf("Matrix Multiplication - Size: %dx%d\n", SIZE, SIZE);
    printf("Initializing matrices...\n");
    
    initialize_matrices();
    
    printf("Starting matrix multiplication...\n");
    
    double start_time = get_time_ms();
    
    matrix_multiply();
    
    double end_time = get_time_ms();
    double elapsed_ms = end_time - start_time;
    
    printf("Time: %.1f ms\n", elapsed_ms);
    
    double checksum = compute_checksum();
    printf("Checksum: %.2f\n", checksum);
    printf("Sample values: C[0][0]=%.2f, C[%d][%d]=%.2f\n", 
           C[0][0], SIZE/2, SIZE/2, C[SIZE/2][SIZE/2]);
    
    return 0;
}