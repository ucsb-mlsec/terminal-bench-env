#include <iostream>

// Forward declarations from mathlib
int add(int a, int b);
int multiply(int a, int b);
int square(int x);

int main() {
    std::cout << "Math Library Calculator Demo\n";
    std::cout << "=============================\n\n";
    
    int a = 5, b = 3;
    int result_add = add(a, b);
    std::cout << "add(" << a << ", " << b << ") = " << result_add << "\n";
    
    int c = 4, d = 7;
    int result_multiply = multiply(c, d);
    std::cout << "multiply(" << c << ", " << d << ") = " << result_multiply << "\n";
    
    int e = 6;
    int result_square = square(e);
    std::cout << "square(" << e << ") = " << result_square << "\n";
    
    std::cout << "\nAll calculations completed successfully!\n";
    
    return 0;
}