#include "mylib.h"
#include <iostream>

namespace MyLib {

int add(int a, int b) {
    return a + b;
}

void printMessage(const std::string& message) {
    std::cout << "MyLib says: " << message << std::endl;
}

Calculator::Calculator() : value_(0) {
}

void Calculator::setValue(int value) {
    value_ = value;
}

int Calculator::getValue() const {
    return value_;
}

} // namespace MyLib