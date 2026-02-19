#include "matrix.hpp"
#include "operations.hpp"
#include <iostream>

int main() {
    // Create test matrices
    Matrix<double> a(2, 2, 1.0);  // all ones
    Matrix<double> b(2, 2, 2.0);  // all twos
    Matrix<double> c(2, 2, 3.0);  // all threes
    Matrix<double> d(2, 2, 4.0);  // all fours
    Matrix<double> e(2, 2, 5.0);  // all fives

    // Wrap them in MatrixExpr
    auto ea = MatrixExpr(a);
    auto eb = MatrixExpr(b);
    auto ec = MatrixExpr(c);
    auto ed = MatrixExpr(d);
    auto ee = MatrixExpr(e);

    // Create a long expression chain that will cause template depth issues
    auto expr = ea + eb + ec + ed + ee;

    // Evaluate the expression
    auto result = evaluate(expr);

    // Print the result
    std::cout << "Result of a+b+c+d+e:\n";
    result.print();

    // Verify correctness (all values should be 15.0)
    bool passed = true;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (result(i, j) != 15.0) {
                passed = false;
                std::cout << "Error: Expected 15.0 at (" << i << "," << j << "), got " << result(i, j) << "\n";
            }
        }
    }

    if (passed) {
        std::cout << "Test passed!\n";
        return 0;
    } else {
        std::cout << "Test failed!\n";
        return 1;
    }
}