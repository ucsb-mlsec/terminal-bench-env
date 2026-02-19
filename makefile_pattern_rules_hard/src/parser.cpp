#include "parser.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

namespace parser {

bool parseData(const std::string& data) {
    if (data.empty()) {
        std::cerr << "Error: Empty data provided to parser" << std::endl;
        return false;
    }
    
#ifdef DEBUG
    std::cout << "Parsing data in DEBUG mode: " << data << std::endl;
#endif
    
    // Simulate parsing logic
    return data.length() > 0;
}

std::vector<std::string> tokenize(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    
    while (stream >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

void printTokens(const std::vector<std::string>& tokens) {
    std::cout << "Tokens: ";
    for (const auto& token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
}

}