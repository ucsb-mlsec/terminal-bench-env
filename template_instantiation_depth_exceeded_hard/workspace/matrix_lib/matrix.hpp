#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <cstddef>

template<typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data;
    size_t m_rows;
    size_t m_cols;

public:
    Matrix(size_t rows, size_t cols, T init_val = T())
        : data(rows, std::vector<T>(cols, init_val)), m_rows(rows), m_cols(cols) {}

    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }

    T& operator()(size_t i, size_t j) {
        return data[i][j];
    }

    const T& operator()(size_t i, size_t j) const {
        return data[i][j];
    }

    void print() const {
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Expression template base class using CRTP
template<typename E, typename T>
class Expr {
public:
    size_t rows() const {
        return static_cast<const E&>(*this).rows();
    }

    size_t cols() const {
        return static_cast<const E&>(*this).cols();
    }

    T operator()(size_t i, size_t j) const {
        return static_cast<const E&>(*this)(i, j);
    }

    const E& derived() const {
        return static_cast<const E&>(*this);
    }
};

// Wrapper for Matrix to make it an expression
template<typename T>
class MatrixExpr : public Expr<MatrixExpr<T>, T> {
private:
    const Matrix<T>& mat;

public:
    MatrixExpr(const Matrix<T>& m) : mat(m) {}

    size_t rows() const { return mat.rows(); }
    size_t cols() const { return mat.cols(); }
    T operator()(size_t i, size_t j) const { return mat(i, j); }
};

// Operation structs
struct AddOp {
    template<typename T>
    static T apply(T a, T b) {
        return a + b;
    }
};

struct MulOp {
    template<typename T>
    static T apply(T a, T b) {
        return a * b;
    }
};

// Binary expression template - THE BUG IS HERE
// Storing L and R by VALUE causes recursive copying and deep template nesting
template<typename Op, typename L, typename R, typename T>
class BinaryExpr : public Expr<BinaryExpr<Op, L, R, T>, T> {
private:
    L left;  // BUG: Stored by value, not reference
    R right; // BUG: Stored by value, not reference

public:
    BinaryExpr(L l, R r) : left(l), right(r) {}

    size_t rows() const { return left.rows(); }
    size_t cols() const { return left.cols(); }

    T operator()(size_t i, size_t j) const {
        return Op::apply(left(i, j), right(i, j));
    }
};

// Operator overloads - these pass expressions by value, triggering the bug
template<typename E1, typename E2, typename T>
auto operator+(const Expr<E1, T>& a, const Expr<E2, T>& b) {
    return BinaryExpr<AddOp, E1, E2, T>(a.derived(), b.derived());
}

template<typename E1, typename E2, typename T>
auto operator*(const Expr<E1, T>& a, const Expr<E2, T>& b) {
    return BinaryExpr<MulOp, E1, E2, T>(a.derived(), b.derived());
}

// Evaluate expression into concrete Matrix
template<typename E, typename T>
Matrix<T> evaluate(const Expr<E, T>& expr) {
    Matrix<T> result(expr.rows(), expr.cols());
    for (size_t i = 0; i < expr.rows(); ++i) {
        for (size_t j = 0; j < expr.cols(); ++j) {
            result(i, j) = expr(i, j);
        }
    }
    return result;
}

#endif // MATRIX_HPP