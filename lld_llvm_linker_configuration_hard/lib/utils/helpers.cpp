#include "helpers.h"
#include <iostream>
#include <string>

void print_result(const std::string& operation, int result) {
    std::cout << "Operation: " << operation << " = " << result << std::endl;
}

std::string format_message(const std::string& msg) {
    return "[Calculator] " + msg;
}

void print_banner() {
    std::cout << "================================" << std::endl;
    std::cout << "  Mathematical Calculator v1.0  " << std::endl;
    std::cout << "================================" << std::endl;
}

void print_separator() {
    std::cout << "--------------------------------" << std::endl;
}