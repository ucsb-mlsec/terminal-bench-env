#include <iostream>
#include <vector>
#include <omp.h>
#include "matrix_ops.h"

int main() {
    std::cout << "Matrix Computation Application" << std::endl;
    std::cout << "Initializing parallel processing..." << std::endl;
    
    // Display OpenMP configuration
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "Number of threads available: " << omp_get_num_threads() << std::endl;
        }
    }
    
    // Define matrix dimensions
    const int rows = 200;
    const int cols = 200;
    
    std::cout << "Creating matrices of size " << rows << "x" << cols << std::endl;
    
    // Create and initialize first matrix
    std::vector<std::vector<double>> matrixA(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrixA[i][j] = i + j * 0.1;
        }
    }
    
    // Create and initialize second matrix
    std::vector<std::vector<double>> matrixB(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrixB[i][j] = i * 0.5 + j;
        }
    }
    
    std::cout << "Performing matrix multiplication..." << std::endl;
    
    // Perform matrix multiplication
    std::vector<std::vector<double>> result = multiplyMatrices(matrixA, matrixB);
    
    // Verify result
    if (!result.empty() && result.size() == rows && result[0].size() == cols) {
        std::cout << "Matrix multiplication completed successfully!" << std::endl;
        std::cout << "Result matrix dimensions: " << result.size() << "x" << result[0].size() << std::endl;
        std::cout << "Sample result value [0][0]: " << result[0][0] << std::endl;
    } else {
        std::cerr << "Error: Matrix multiplication produced invalid result" << std::endl;
        return 1;
    }
    
    return 0;
}