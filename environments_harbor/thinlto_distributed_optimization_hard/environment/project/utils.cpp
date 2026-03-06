#include <iostream>
#include <string>
#include <vector>
#include "utils.h"

namespace Utils {

bool initializeUtils() {
    std::cout << "Initializing utility module..." << std::endl;
    return true;
}

std::string formatString(const std::string& input, bool uppercase) {
    std::string result = input;
    if (uppercase) {
        for (char& c : result) {
            c = std::toupper(static_cast<unsigned char>(c));
        }
    } else {
        for (char& c : result) {
            c = std::tolower(static_cast<unsigned char>(c));
        }
    }
    return result;
}

std::vector<int> convertData(const std::vector<std::string>& stringData) {
    std::vector<int> result;
    result.reserve(stringData.size());
    
    for (const auto& str : stringData) {
        try {
            int value = std::stoi(str);
            result.push_back(value);
        } catch (const std::exception& e) {
            std::cerr << "Error converting: " << str << std::endl;
            result.push_back(0);
        }
    }
    
    return result;
}

} // namespace Utils