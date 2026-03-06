#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"

const double MATRIX_EPSILON = 1e-10;

class Matrix {
private:
    double** data;
    int rows;
    int cols;

public:
    Matrix(int r, int c) : rows(r), cols(c) {
        data = new double*[rows];
        for (int i = 0; i < rows; i++) {
            data[i] = new double[cols];
            for (int j = 0; j < cols; j++) {
                data[i][j] = 0.0;
            }
        }
    }

    ~Matrix() {
        for (int i = 0; i < rows; i++) {
            delete[] data[i];
        }
        delete[] data;
    }

    double get(int i, int j) const {
        return data[i][j];
    }

    void set(int i, int j, double val) {
        data[i][j] = val;
    }

    int getRows() const {
        return rows;
    }

    int getCols() const {
        return cols;
    }
};

Matrix multiply(const Matrix& a, const Matrix& b) {
    int aRows = a.getRows();
    int aCols = a.getCols();
    int bRows = b.getRows();
    int bCols = b.getCols();
    
    Matrix result(aRows, bCols);
    
    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bCols; j++) {
            double sum = 0.0;
            for (int k = 0; k < aCols; k++) {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
    
    return result;
}

#endif