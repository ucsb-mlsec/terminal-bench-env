#include <iostream>
#include <fstream>

int main() {
    std::ofstream outfile("generated.h");
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open generated.h for writing" << std::endl;
        return 1;
    }
    
    outfile << "#ifndef GENERATED_H\n";
    outfile << "#define GENERATED_H\n";
    outfile << "\n";
    outfile << "#define MAGIC_NUMBER 42\n";
    outfile << "\n";
    outfile << "inline int getMagicNumber() { return MAGIC_NUMBER; }\n";
    outfile << "\n";
    outfile << "#endif\n";
    
    outfile.close();
    
    std::cout << "Generated generated.h successfully" << std::endl;
    
    return 0;
}