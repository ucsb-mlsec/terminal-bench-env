#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include "matrix.hpp"
#include <cassert>

template<typename T>
Matrix<T> transpose(const Matrix<T>& input) {
    Matrix<T> result(input.cols(), input.rows());
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            result(j, i) = input(i, j);
        }
    }
    return result;
}

template<typename T>
Matrix<T> matmul(const Matrix<T>& a, const Matrix<T>& b) {
    assert(a.cols() == b.rows() && "Matrix dimensions must match for multiplication");
    
    Matrix<T> result(a.rows(), b.cols());
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < b.cols(); ++j) {
            T sum = T(0);
            for (size_t k = 0; k < a.cols(); ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

template<typename T>
Matrix<T> scale(const Matrix<T>& m, T scalar) {
    Matrix<T> result(m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            result(i, j) = m(i, j) * scalar;
        }
    }
    return result;
}

template<typename T>
Matrix<T> identity(size_t n) {
    Matrix<T> result(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result(i, j) = (i == j) ? T(1) : T(0);
        }
    }
    return result;
}

#endif