#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <vector>

// Multiply two matrices using parallel processing
std::vector<std::vector<double>> multiplyMatrices(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B
);

#endif // MATRIX_OPS_H