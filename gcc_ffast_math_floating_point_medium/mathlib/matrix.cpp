#include <cmath>
#include <vector>
#include <cstddef>

// Matrix multiplication: C = A * B
void matrix_multiply(const std::vector<std::vector<double>>& A, 
                     const std::vector<std::vector<double>>& B, 
                     std::vector<std::vector<double>>& C) {
    size_t rows_a = A.size();
    size_t cols_a = A[0].size();
    size_t cols_b = B[0].size();
    
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_b; ++j) {
            C[i][j] = 0.0;
            for (size_t k = 0; k < cols_a; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Matrix transpose: B = A^T
void matrix_transpose(const std::vector<std::vector<double>>& A, 
                      std::vector<std::vector<double>>& B) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            B[j][i] = A[i][j];
        }
    }
}

// Matrix scaling with normalization
void matrix_normalize(std::vector<std::vector<double>>& A) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    
    // Calculate Frobenius norm
    double norm = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            norm += A[i][j] * A[i][j];
        }
    }
    norm = sqrt(norm);
    
    // Normalize matrix
    if (norm > 0.0) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                A[i][j] /= norm;
            }
        }
    }
}

// Element-wise matrix exponential
void matrix_exp(const std::vector<std::vector<double>>& A, 
                std::vector<std::vector<double>>& B) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            B[i][j] = exp(A[i][j]);
        }
    }
}