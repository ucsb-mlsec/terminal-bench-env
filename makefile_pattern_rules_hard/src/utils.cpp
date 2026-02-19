#include "utils.hpp"
#include <iostream>
#include <string>

void printWelcome() {
    std::cout << "Welcome to the Multi-Config Build System Demo!" << std::endl;
#ifdef DEBUG
    std::cout << "Running in DEBUG mode" << std::endl;
#endif
#ifdef NDEBUG
    std::cout << "Running in RELEASE mode" << std::endl;
#endif
}

std::string formatString(const std::string& input) {
    return "[" + input + "]";
}

int calculateSum(int a, int b) {
    return a + b;
}

void printBuildInfo() {
    std::cout << "Build configuration active" << std::endl;
}