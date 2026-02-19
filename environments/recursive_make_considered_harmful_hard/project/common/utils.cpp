#include "utils.h"
#include <string>
#include <sstream>
#include <iostream>

namespace utils {

int calculate(int a, int b, char operation) {
    switch (operation) {
        case '+':
            return a + b;
        case '-':
            return a - b;
        case '*':
            return a * b;
        case '/':
            if (b != 0) {
                return a / b;
            }
            return 0;
        default:
            return 0;
    }
}

std::string formatMessage(const std::string& prefix, const std::string& message) {
    std::ostringstream oss;
    oss << "[" << prefix << "] " << message;
    return oss.str();
}

bool validateInput(int value, int min, int max) {
    if (value < min || value > max) {
        return false;
    }
    return true;
}

} // namespace utils