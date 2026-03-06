#include "utils.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <ctime>

std::string readFileContents(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    return buffer.str();
}

std::string formatString(const std::string& input, int width, char fillChar) {
    std::string result = input;
    
    if (result.length() < static_cast<size_t>(width)) {
        int padding = width - result.length();
        int leftPad = padding / 2;
        int rightPad = padding - leftPad;
        
        result = std::string(leftPad, fillChar) + result + std::string(rightPad, fillChar);
    }
    
    return result;
}

void logMessage(const std::string& message, LogLevel level) {
    std::time_t now = std::time(nullptr);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    
    std::string levelStr;
    switch (level) {
        case INFO:
            levelStr = "INFO";
            break;
        case WARNING:
            levelStr = "WARNING";
            break;
        case ERROR:
            levelStr = "ERROR";
            break;
        default:
            levelStr = "UNKNOWN";
    }
    
    std::cout << "[" << timestamp << "] [" << levelStr << "] " << message << std::endl;
}

ConfigMap parseConfigFile(const std::string& filename) {
    ConfigMap config;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file " << filename << std::endl;
        return config;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        size_t delimPos = line.find('=');
        if (delimPos != std::string::npos) {
            std::string key = line.substr(0, delimPos);
            std::string value = line.substr(delimPos + 1);
            
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            
            config[key] = value;
        }
    }
    
    file.close();
    return config;
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}