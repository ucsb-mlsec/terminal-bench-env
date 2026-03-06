#include <iostream>
#include <string>

void printInfo() {
    std::cout << "Program version 1.0" << std::endl;
    std::cout << "Built with C++" << std::endl;
    std::cout << "Copyright 2024" << std::endl;
    std::cout << "A sample utility program" << std::endl;
}

std::string getVersion() {
    return "1.0.0";
}

void printSeparator() {
    std::cout << "========================================" << std::endl;
}