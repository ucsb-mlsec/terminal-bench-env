#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define BLOCK_SIZE 64

// Matrix structure
typedef struct {
    int **data;
    int size;
} Matrix;

// Allocate matrix
Matrix* allocate_matrix(int size) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->size = size;
    m->data = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; i++) {
        m->data[i] = (int*)calloc(size, sizeof(int));
    }
    return m;
}

// Free matrix
void free_matrix(Matrix *m) {
    for (int i = 0; i < m->size; i++) {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

// Read matrix from file
Matrix* read_matrix(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error opening input file\n");
        exit(1);
    }
    
    int size;
    fscanf(fp, "%d", &size);
    
    Matrix *m = allocate_matrix(size);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fscanf(fp, "%d", &m->data[i][j]);
        }
    }
    
    fclose(fp);
    return m;
}

// Write matrix to file
void write_matrix(const char *filename, Matrix *m) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening output file\n");
        exit(1);
    }
    
    fprintf(fp, "%d\n", m->size);
    for (int i = 0; i < m->size; i++) {
        for (int j = 0; j < m->size; j++) {
            fprintf(fp, "%d", m->data[i][j]);
            if (j < m->size - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
}

// Block-based matrix multiplication with nested parallelism
void block_matrix_multiply(Matrix *A, Matrix *B, Matrix *C) {
    int n = A->size;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Outer parallel region - processes blocks
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int bi = 0; bi < num_blocks; bi++) {
        for (int bj = 0; bj < num_blocks; bj++) {
            int block_start_i = bi * BLOCK_SIZE;
            int block_start_j = bj * BLOCK_SIZE;
            int block_end_i = (block_start_i + BLOCK_SIZE > n) ? n : block_start_i + BLOCK_SIZE;
            int block_end_j = (block_start_j + BLOCK_SIZE > n) ? n : block_start_j + BLOCK_SIZE;
            
            // Inner parallel region - processes within each block
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = block_start_i; i < block_end_i; i++) {
                for (int j = block_start_j; j < block_end_j; j++) {
                    long long sum = 0;
                    for (int k = 0; k < n; k++) {
                        sum += (long long)A->data[i][k] * B->data[k][j];
                    }
                    C->data[i][j] = (int)(sum % 1000000);
                }
            }
        }
    }
}

// Additional computation to increase workload
void transform_matrix(Matrix *m) {
    int n = m->size;
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n; j++) {
            int val = m->data[i][j];
            // Some computation
            for (int k = 0; k < 10; k++) {
                val = (val * 13 + 7) % 1000000;
            }
            m->data[i][j] = val;
        }
    }
}

int main() {
    double start_time, end_time;
    
    start_time = omp_get_wtime();
    
    // Read input matrix
    Matrix *A = read_matrix("/workspace/compute/input.txt");
    
    // Create matrices for computation
    Matrix *B = allocate_matrix(A->size);
    Matrix *C = allocate_matrix(A->size);
    
    // Initialize B as transpose of A
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A->size; i++) {
        for (int j = 0; j < A->size; j++) {
            B->data[i][j] = A->data[j][i];
        }
    }
    
    // Perform block-based matrix multiplication (nested parallelism)
    block_matrix_multiply(A, B, C);
    
    // Additional transformation (nested parallelism)
    transform_matrix(C);
    
    // Write result
    write_matrix("/workspace/compute/output.txt", C);
    
    end_time = omp_get_wtime();
    
    printf("Computation completed in %.2f seconds\n", end_time - start_time);
    printf("Matrix size: %d x %d\n", A->size, A->size);
    
    // Cleanup
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    
    return 0;
}