#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * matrix_multiply - Multiplies two matrices
 * @a: First input matrix (m x n)
 * @b: Second input matrix (n x p)
 * @result: Output matrix (m x p)
 * @m: Number of rows in matrix a
 * @n: Number of columns in a / rows in b
 * @p: Number of columns in matrix b
 * 
 * Performs C = A * B where A is m x n and B is n x p
 */
void matrix_multiply(const double *a, const double *b, double *result, 
                     int m, int n, int p) {
    // Initialize result matrix to zero
    memset(result, 0, m * p * sizeof(double));
    
    // Standard matrix multiplication algorithm
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                result[i * p + j] += a[i * n + k] * b[k * p + j];
            }
        }
    }
}

/**
 * matrix_add - Adds two matrices element-wise
 * @a: First input matrix
 * @b: Second input matrix
 * @result: Output matrix
 * @rows: Number of rows
 * @cols: Number of columns
 * 
 * Performs C = A + B where all matrices are rows x cols
 */
void matrix_add(const double *a, const double *b, double *result, 
                int rows, int cols) {
    int total_elements = rows * cols;
    
    // Element-wise addition
    for (int i = 0; i < total_elements; i++) {
        result[i] = a[i] + b[i];
    }
}

/**
 * matrix_subtract - Subtracts two matrices element-wise
 * @a: First input matrix
 * @b: Second input matrix
 * @result: Output matrix
 * @rows: Number of rows
 * @cols: Number of columns
 * 
 * Performs C = A - B where all matrices are rows x cols
 */
void matrix_subtract(const double *a, const double *b, double *result,
                     int rows, int cols) {
    int total_elements = rows * cols;
    
    // Element-wise subtraction
    for (int i = 0; i < total_elements; i++) {
        result[i] = a[i] - b[i];
    }
}

/**
 * matrix_transpose - Transposes a matrix
 * @input: Input matrix (rows x cols)
 * @output: Output matrix (cols x rows)
 * @rows: Number of rows in input matrix
 * @cols: Number of columns in input matrix
 * 
 * Converts rows to columns and columns to rows
 */
void matrix_transpose(const double *input, double *output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

/**
 * matrix_identity - Creates an identity matrix
 * @matrix: Output matrix (must be preallocated)
 * @size: Dimension of the square matrix
 * 
 * Creates a size x size identity matrix with 1s on diagonal, 0s elsewhere
 */
void matrix_identity(double *matrix, int size) {
    // Initialize all elements to zero
    memset(matrix, 0, size * size * sizeof(double));
    
    // Set diagonal elements to 1
    for (int i = 0; i < size; i++) {
        matrix[i * size + i] = 1.0;
    }
}

/**
 * matrix_scale - Scales a matrix by a scalar value
 * @input: Input matrix
 * @output: Output matrix
 * @rows: Number of rows
 * @cols: Number of columns
 * @scalar: Scaling factor
 * 
 * Multiplies each element of the matrix by the scalar
 */
void matrix_scale(const double *input, double *output, int rows, int cols, 
                  double scalar) {
    int total_elements = rows * cols;
    
    for (int i = 0; i < total_elements; i++) {
        output[i] = input[i] * scalar;
    }
}

/**
 * matrix_dot_product - Computes dot product of two vectors treated as matrices
 * @a: First vector
 * @b: Second vector
 * @length: Length of vectors
 * 
 * Returns the dot product of two vectors
 */
double matrix_dot_product(const double *a, const double *b, int length) {
    double sum = 0.0;
    
    for (int i = 0; i < length; i++) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

/**
 * matrix_fill - Fills a matrix with a specific value
 * @matrix: Matrix to fill
 * @rows: Number of rows
 * @cols: Number of columns
 * @value: Value to fill with
 */
void matrix_fill(double *matrix, int rows, int cols, double value) {
    int total_elements = rows * cols;
    
    for (int i = 0; i < total_elements; i++) {
        matrix[i] = value;
    }
}

/**
 * matrix_copy - Copies one matrix to another
 * @source: Source matrix
 * @dest: Destination matrix
 * @rows: Number of rows
 * @cols: Number of columns
 */
void matrix_copy(const double *source, double *dest, int rows, int cols) {
    int total_elements = rows * cols;
    memcpy(dest, source, total_elements * sizeof(double));
}

/**
 * matrix_element_multiply - Element-wise multiplication of two matrices
 * @a: First input matrix
 * @b: Second input matrix
 * @result: Output matrix
 * @rows: Number of rows
 * @cols: Number of columns
 * 
 * Also known as Hadamard product
 */
void matrix_element_multiply(const double *a, const double *b, double *result,
                             int rows, int cols) {
    int total_elements = rows * cols;
    
    for (int i = 0; i < total_elements; i++) {
        result[i] = a[i] * b[i];
    }
}

/**
 * matrix_sum - Computes the sum of all elements in a matrix
 * @matrix: Input matrix
 * @rows: Number of rows
 * @cols: Number of columns
 * 
 * Returns the sum of all matrix elements
 */
double matrix_sum(const double *matrix, int rows, int cols) {
    int total_elements = rows * cols;
    double sum = 0.0;
    
    for (int i = 0; i < total_elements; i++) {
        sum += matrix[i];
    }
    
    return sum;
}