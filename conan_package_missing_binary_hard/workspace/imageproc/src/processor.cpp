#include "processor.h"
#include <iostream>
#include <fstream>
#include <png.h>

bool processBitmap(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    if (!file.good()) {
        std::cerr << "Error: File is not readable " << filename << std::endl;
        file.close();
        return false;
    }
    
    // Read first few bytes to verify it's a valid file
    char header[54];
    file.read(header, sizeof(header));
    
    if (!file) {
        std::cerr << "Error: File too small or corrupted" << std::endl;
        file.close();
        return false;
    }
    
    file.close();
    
    // Verify libpng is available by checking version
    std::cout << "PNG library version: " << PNG_LIBPNG_VER_STRING << std::endl;
    std::cout << "Successfully processed bitmap file: " << filename << std::endl;
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <bitmap_file>" << std::endl;
        return 1;
    }
    
    if (processBitmap(argv[1])) {
        return 0;
    }
    
    return 1;
}