#include <iostream>
#include <zlib.h>
#include <cstring>

int main() {
    std::cout << "Math Processor - Using zlib compression" << std::endl;
    
    // Display zlib version
    std::cout << "zlib version: " << zlibVersion() << std::endl;
    
    // Original data to compress
    const char* original = "This is a sample text for compression testing in the math processor application.";
    uLong originalSize = strlen(original) + 1;
    
    std::cout << "Original data: " << original << std::endl;
    std::cout << "Original size: " << originalSize << " bytes" << std::endl;
    
    // Prepare buffer for compressed data
    uLong compressedSize = compressBound(originalSize);
    unsigned char* compressedData = new unsigned char[compressedSize];
    
    // Compress the data
    int result = compress(compressedData, &compressedSize, 
                         (const unsigned char*)original, originalSize);
    
    if (result == Z_OK) {
        std::cout << "Compression successful!" << std::endl;
        std::cout << "Compressed size: " << compressedSize << " bytes" << std::endl;
        std::cout << "Compression ratio: " 
                  << (100.0 * compressedSize / originalSize) << "%" << std::endl;
        
        // Clean up
        delete[] compressedData;
        
        std::cout << "Math processor initialized successfully with compression support." << std::endl;
        return 0;
    } else {
        std::cerr << "Compression failed with error code: " << result << std::endl;
        delete[] compressedData;
        return 1;
    }
}