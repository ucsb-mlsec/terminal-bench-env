#include <iostream>
#include "Calculator.h"

using namespace std;

int main() {
    Calculator calc;
    
    // Test addition
    cout << "10 + 5 = " << calc.add(10, 5) << endl;
    
    // Test subtraction
    cout << "20 - 8 = " << calc.subtract(20, 8) << endl;
    
    // Test multiplication
    cout << "4 * 7 = " << calc.multiply(4, 7) << endl;
    
    // Test division
    cout << "100 / 4 = " << calc.divide(100, 4) << endl;
    
    // Test division by zero
    cout << "10 / 0 = " << calc.divide(10, 0) << endl;
    
    // Display operation count
    cout << "Total operations performed: " << calc.getOperationCount() << endl;
    
    // Test reset
    calc.reset();
    cout << "Calculator reset" << endl;
    cout << "Operations after reset: " << calc.getOperationCount() << endl;
    
    // Perform operation after reset
    cout << "5 + 3 = " << calc.add(5, 3) << endl;
    cout << "Operations after adding: " << calc.getOperationCount() << endl;
    
    return 0;
}