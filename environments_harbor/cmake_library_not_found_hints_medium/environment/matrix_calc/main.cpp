#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
    cout << "Matrix Calculator - Eigen Library Demo" << endl;
    cout << "=======================================" << endl << endl;

    // Create a 3x3 matrix A
    Matrix3d A;
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    
    cout << "Matrix A:" << endl;
    cout << A << endl << endl;

    // Create a 3x3 matrix B
    Matrix3d B;
    B << 9, 8, 7,
         6, 5, 4,
         3, 2, 1;
    
    cout << "Matrix B:" << endl;
    cout << B << endl << endl;

    // Matrix addition
    Matrix3d C = A + B;
    cout << "A + B:" << endl;
    cout << C << endl << endl;

    // Matrix multiplication
    Matrix3d D = A * B;
    cout << "A * B:" << endl;
    cout << D << endl << endl;

    // Matrix transpose
    Matrix3d E = A.transpose();
    cout << "Transpose of A:" << endl;
    cout << E << endl << endl;

    // Create a dynamic size matrix
    MatrixXd F(2, 3);
    F << 1.5, 2.5, 3.5,
         4.5, 5.5, 6.5;
    
    cout << "Matrix F (2x3):" << endl;
    cout << F << endl << endl;

    // Scalar multiplication
    MatrixXd G = F * 2.0;
    cout << "F * 2.0:" << endl;
    cout << G << endl << endl;

    cout << "All matrix operations completed successfully!" << endl;

    return 0;
}