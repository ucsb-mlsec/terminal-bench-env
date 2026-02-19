#include "core.hpp"
#include <iostream>

namespace Core {

CoreBase::CoreBase() {
    std::cout << "CoreBase constructor called" << std::endl;
}

CoreBase::~CoreBase() {
    std::cout << "CoreBase destructor called" << std::endl;
}

void CoreBase::execute() {
    std::cout << "CoreBase::execute() - Base implementation" << std::endl;
}

std::string CoreBase::getName() const {
    return "CoreBase";
}

std::string getCoreVersion() {
    return "Core v1.0";
}

void initializeCore() {
    std::cout << "Core initialized - Version: " << getCoreVersion() << std::endl;
}

} // namespace Core