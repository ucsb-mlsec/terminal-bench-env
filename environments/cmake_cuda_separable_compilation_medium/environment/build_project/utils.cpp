#include "utils.h"
#include <iostream>

void printWelcome() {
    std::cout << "Welcome to the Build Project!" << std::endl;
    std::cout << "This is a demonstration of CMake build system." << std::endl;
}

std::string getVersion() {
    return "1.0.0";
}

void printMessage(const std::string& msg) {
    std::cout << "Message: " << msg << std::endl;
}