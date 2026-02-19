#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MATRIX_SIZE 2000
#define OUTPUT_FILE "matrix_data.bin"

int main() {
    double *matrix;
    FILE *fp;
    size_t elements_written;
    size_t total_elements = MATRIX_SIZE * MATRIX_SIZE;
    
    // Seed random number generator for reproducibility
    srand(12345);
    
    // Allocate memory for matrix
    matrix = (double *)malloc(total_elements * sizeof(double));
    if (matrix == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for matrix\n");
        return 1;
    }
    
    printf("Generating %dx%d matrix...\n", MATRIX_SIZE, MATRIX_SIZE);
    
    // Fill matrix with random values between 0.0 and 100.0
    for (size_t i = 0; i < total_elements; i++) {
        matrix[i] = ((double)rand() / RAND_MAX) * 100.0;
    }
    
    // Open file for binary writing
    fp = fopen(OUTPUT_FILE, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file %s for writing\n", OUTPUT_FILE);
        free(matrix);
        return 1;
    }
    
    // Write matrix to file
    elements_written = fwrite(matrix, sizeof(double), total_elements, fp);
    if (elements_written != total_elements) {
        fprintf(stderr, "Error: Failed to write complete matrix to file\n");
        fclose(fp);
        free(matrix);
        return 1;
    }
    
    // Close file
    fclose(fp);
    
    // Calculate and print file size
    double file_size_mb = (total_elements * sizeof(double)) / (1024.0 * 1024.0);
    printf("Successfully generated %s (%.2f MB)\n", OUTPUT_FILE, file_size_mb);
    printf("Matrix dimensions: %d x %d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("Total elements: %zu\n", total_elements);
    
    // Free allocated memory
    free(matrix);
    
    return 0;
}