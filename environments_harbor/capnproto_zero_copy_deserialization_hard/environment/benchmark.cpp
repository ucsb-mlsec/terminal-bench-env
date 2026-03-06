#include <capnp/message.h>
#include <capnp/serialize.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "message.capnp.h"

// Base64 decoding lookup table
static const unsigned char base64_decode_table[256] = {
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
    64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
    64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
};

std::vector<unsigned char> decodeBase64(const std::string& encoded) {
    std::vector<unsigned char> decoded;
    std::vector<unsigned char> temp;
    
    for (char c : encoded) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        unsigned char val = base64_decode_table[static_cast<unsigned char>(c)];
        if (val < 64) {
            temp.push_back(val);
        }
    }
    
    for (size_t i = 0; i + 3 < temp.size(); i += 4) {
        decoded.push_back((temp[i] << 2) | (temp[i + 1] >> 4));
        decoded.push_back((temp[i + 1] << 4) | (temp[i + 2] >> 2));
        decoded.push_back((temp[i + 2] << 6) | temp[i + 3]);
    }
    
    size_t remaining = temp.size() % 4;
    size_t i = temp.size() - remaining;
    if (remaining == 2) {
        decoded.push_back((temp[i] << 2) | (temp[i + 1] >> 4));
    } else if (remaining == 3) {
        decoded.push_back((temp[i] << 2) | (temp[i + 1] >> 4));
        decoded.push_back((temp[i + 1] << 4) | (temp[i + 2] >> 2));
    }
    
    return decoded;
}

std::vector<unsigned char> loadBinaryData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::string base64Content;
    std::getline(file, base64Content);
    file.close();
    
    return decodeBase64(base64Content);
}

// Implementation A: Process data directly from the message
// BUG: This implementation is accessing the list multiple times in inefficient ways
int64_t processDirectly(const std::vector<unsigned char>& data) {
    kj::ArrayPtr<const capnp::word> words(
        reinterpret_cast<const capnp::word*>(data.data()),
        data.size() / sizeof(capnp::word)
    );
    
    capnp::FlatArrayMessageReader message(words);
    auto numberList = message.getRoot<NumberList>();
    auto numbers = numberList.getNumbers();
    
    int64_t sum = 0;
    // BUG: Calling .size() in the loop condition causes it to be evaluated every iteration
    // This forces repeated access to the message structure
    for (unsigned int i = 0; i < numbers.size(); i++) {
        sum += numbers[i];
    }
    
    return sum;
}

// Implementation B: Extract all data into a vector first, then process
int64_t processWithExtraction(const std::vector<unsigned char>& data) {
    kj::ArrayPtr<const capnp::word> words(
        reinterpret_cast<const capnp::word*>(data.data()),
        data.size() / sizeof(capnp::word)
    );
    
    capnp::FlatArrayMessageReader message(words);
    auto numberList = message.getRoot<NumberList>();
    auto numbers = numberList.getNumbers();
    
    // Extract all values into a vector
    std::vector<int32_t> extractedNumbers;
    extractedNumbers.reserve(numbers.size());
    for (auto num : numbers) {
        extractedNumbers.push_back(num);
    }
    
    // Now process from the vector
    int64_t sum = 0;
    for (int32_t num : extractedNumbers) {
        sum += num;
    }
    
    return sum;
}

int main() {
    try {
        // Load and decode test data
        std::cout << "Loading test data..." << std::endl;
        auto binaryData = loadBinaryData("/workspace/test_data.txt");
        std::cout << "Loaded " << binaryData.size() << " bytes" << std::endl;
        
        const int iterations = 1000;
        
        // Warm up
        processDirectly(binaryData);
        processWithExtraction(binaryData);
        
        // Benchmark Implementation A
        std::cout << "Benchmarking Implementation A (direct access)..." << std::endl;
        auto startA = std::chrono::high_resolution_clock::now();
        int64_t sumA = 0;
        for (int i = 0; i < iterations; i++) {
            sumA = processDirectly(binaryData);
        }
        auto endA = std::chrono::high_resolution_clock::now();
        auto durationA = std::chrono::duration_cast<std::chrono::milliseconds>(endA - startA).count();
        
        // Benchmark Implementation B
        std::cout << "Benchmarking Implementation B (extract then process)..." << std::endl;
        auto startB = std::chrono::high_resolution_clock::now();
        int64_t sumB = 0;
        for (int i = 0; i < iterations; i++) {
            sumB = processWithExtraction(binaryData);
        }
        auto endB = std::chrono::high_resolution_clock::now();
        auto durationB = std::chrono::duration_cast<std::chrono::milliseconds>(endB - startB).count();
        
        // Verify results match
        if (sumA != sumB) {
            std::cerr << "ERROR: Sums don't match! A=" << sumA << ", B=" << sumB << std::endl;
            return 1;
        }
        
        std::cout << "\nResults (" << iterations << " iterations):" << std::endl;
        std::cout << "Implementation A: " << durationA << " ms (sum=" << sumA << ")" << std::endl;
        std::cout << "Implementation B: " << durationB << " ms (sum=" << sumB << ")" << std::endl;
        std::cout << "Ratio: " << (double)durationB / durationA << "x" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}