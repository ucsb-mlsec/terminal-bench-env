#include <iostream>
#include <cstring>

class Calculator {
private:
    int* results;
    int capacity;
    int count;

public:
    Calculator(int size) : capacity(size), count(0) {
        results = new int[capacity];
    }

    ~Calculator() {
        delete[] results;
    }

    int add(int a, int b) {
        int result = a + b;
        // Intentional buffer overflow - no bounds checking
        results[count++] = result;
        return result;
    }

    int subtract(int a, int b) {
        return a - b;
    }

    int multiply(int a, int b) {
        return a * b;
    }

    int divide(int a, int b) {
        if (b == 0) {
            std::cerr << "Error: Division by zero" << std::endl;
            return 0;
        }
        return a / b;
    }

    void printResults() {
        for (int i = 0; i < count; i++) {
            std::cout << "Result " << i << ": " << results[i] << std::endl;
        }
    }
};

int main() {
    Calculator calc(5);
    
    std::cout << "Calculator Program" << std::endl;
    std::cout << "5 + 3 = " << calc.add(5, 3) << std::endl;
    std::cout << "10 - 4 = " << calc.subtract(10, 4) << std::endl;
    std::cout << "6 * 7 = " << calc.multiply(6, 7) << std::endl;
    std::cout << "20 / 4 = " << calc.divide(20, 4) << std::endl;
    
    return 0;
}