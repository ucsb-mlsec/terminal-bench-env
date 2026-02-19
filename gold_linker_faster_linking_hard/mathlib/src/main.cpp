#include <iostream>
#include "math_ops.h"

using namespace std;

int main(int argc, char** argv) {
    if (argc != 4) {
        cout << "Usage: calculator <num1> <operator> <num2>" << endl;
        return 1;
    }

    int num1 = atoi(argv[1]);
    string op = argv[2];
    int num2 = atoi(argv[3]);

    int result;

    if (op == "+") {
        result = add(num1, num2);
    } else if (op == "-") {
        result = subtract(num1, num2);
    } else if (op == "*") {
        result = multiply(num1, num2);
    } else if (op == "/") {
        result = divide(num1, num2);
    } else {
        cout << "Unknown operator: " << op << endl;
        return 1;
    }

    cout << result << endl;
    return 0;
}