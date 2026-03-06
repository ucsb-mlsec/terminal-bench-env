#include "matrix.h"
#include "constants.h"
#include "utils.h"
#include <cstring>

Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    // Allocate 2D data array
    data = new double*[rows];
    for (int i = 0; i < rows; i++) {
        data[i] = new double[cols];
        // Initialize to 0.0
        for (int j = 0; j < cols; j++) {
            data[i][j] = 0.0;
        }
    }
    
    // Use some utility function and constant
    if (isValidDimension(rows) && isValidDimension(cols)) {
        // Matrix dimensions are valid
        double init_val = MATRIX_INIT_VALUE;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = init_val;
            }
        }
    }
}

Matrix::~Matrix() {
    // Properly delete 2D data array
    if (data != nullptr) {
        for (int i = 0; i < rows; i++) {
            delete[] data[i];
        }
        delete[] data;
        data = nullptr;
    }
}

double Matrix::get(int i, int j) const {
    return data[i][j];
}

void Matrix::set(int i, int j, double val) {
    data[i][j] = val;
}

int Matrix::getRows() const {
    return rows;
}

int Matrix::getCols() const {
    return cols;
}