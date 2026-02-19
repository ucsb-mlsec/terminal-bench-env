#include <iostream>
#include "plugin.h"
#include "core.h"
#include "utils.h"

int main() {
    std::cout << "=== Testing Multi-Library Symbol Visibility ===" << std::endl;
    std::cout << "This test loads all three libraries together..." << std::endl;
    
    // Test libcore functions
    std::cout << "\n[1] Testing libcore.so..." << std::endl;
    if (core_initialize() != 0) {
        std::cerr << "ERROR: core_initialize() failed!" << std::endl;
        return 1;
    }
    std::cout << "    core_initialize() - OK" << std::endl;
    
    // Test libutils functions
    std::cout << "\n[2] Testing libutils.so..." << std::endl;
    if (utils_setup() != 0) {
        std::cerr << "ERROR: utils_setup() failed!" << std::endl;
        return 1;
    }
    std::cout << "    utils_setup() - OK" << std::endl;
    
    // Test libplugin functions (depends on both libcore and libutils)
    std::cout << "\n[3] Testing libplugin.so..." << std::endl;
    if (plugin_load() != 0) {
        std::cerr << "ERROR: plugin_load() failed!" << std::endl;
        return 1;
    }
    std::cout << "    plugin_load() - OK" << std::endl;
    
    if (plugin_execute() != 0) {
        std::cerr << "ERROR: plugin_execute() failed!" << std::endl;
        return 1;
    }
    std::cout << "    plugin_execute() - OK" << std::endl;
    
    std::cout << "\n=== All Tests PASSED ===" << std::endl;
    std::cout << "Libraries loaded together without symbol conflicts!" << std::endl;
    
    return 0;
}