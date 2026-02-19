#include <fstream>
#include <iostream>

int main() {
    std::ofstream out("generated_constants.h");
    
    if (!out.is_open()) {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }
    
    // Write header guards
    out << "#ifndef GENERATED_CONSTANTS_H" << std::endl;
    out << "#define GENERATED_CONSTANTS_H" << std::endl;
    out << std::endl;
    
    // Write constant definitions
    out << "const int VERSION = 1;" << std::endl;
    out << "const char* BUILD_TYPE = \"release\";" << std::endl;
    out << "const int MAX_BUFFER_SIZE = 4096;" << std::endl;
    out << "const double PI_APPROX = 3.14159;" << std::endl;
    out << std::endl;
    
    // Close header guard
    out << "#endif // GENERATED_CONSTANTS_H" << std::endl;
    
    out.close();
    
    std::cout << "Generated constants header" << std::endl;
    
    return 0;
}