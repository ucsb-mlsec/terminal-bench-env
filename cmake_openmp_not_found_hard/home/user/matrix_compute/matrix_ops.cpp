#include "matrix_ops.h"
#include <vector>
#include <omp.h>
#include <stdexcept>

std::vector<std::vector<double>> multiplyMatrices(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B) {
    
    // Validate input matrices
    if (A.empty() || B.empty()) {
        throw std::invalid_argument("Input matrices cannot be empty");
    }
    
    size_t m = A.size();        // rows of A
    size_t n = A[0].size();     // cols of A
    size_t p = B.size();        // rows of B
    
    if (B.empty() || B[0].empty()) {
        throw std::invalid_argument("Matrix B has invalid dimensions");
    }
    
    size_t q = B[0].size();     // cols of B
    
    // Check if matrices can be multiplied
    if (n != p) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    // Initialize result matrix with zeros
    std::vector<std::vector<double>> C(m, std::vector<double>(q, 0.0));
    
    // Parallel matrix multiplication using OpenMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < q; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    return C;
}