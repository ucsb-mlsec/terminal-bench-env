#include "config.h"
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>

namespace dataproc {

// Singleton instance
Config* Config::instance_ = nullptr;

Config::Config() {
    setDefaults();
}

Config::~Config() {
}

Config& Config::getInstance() {
    if (instance_ == nullptr) {
        instance_ = new Config();
    }
    return *instance_;
}

void Config::setDefaults() {
    settings_["thread_count"] = "4";
    settings_["buffer_size"] = "8192";
    settings_["max_buffer_size"] = "65536";
    settings_["input_path"] = "data/input.csv";
    settings_["output_path"] = "data/output.csv";
    settings_["temp_dir"] = "/tmp/dataproc";
    settings_["log_level"] = "info";
    settings_["enable_compression"] = "false";
    settings_["compression_level"] = "6";
    settings_["chunk_size"] = "1024";
    settings_["max_memory"] = "1073741824";
    settings_["timeout"] = "300";
    settings_["retry_count"] = "3";
    settings_["enable_validation"] = "true";
    settings_["skip_header"] = "true";
    settings_["delimiter"] = ",";
    settings_["quote_char"] = "\"";
    settings_["escape_char"] = "\\";
    settings_["line_ending"] = "auto";
    settings_["encoding"] = "utf-8";
    settings_["enable_cache"] = "true";
    settings_["cache_size"] = "100";
    settings_["worker_queue_size"] = "1000";
    settings_["enable_profiling"] = "false";
}

bool Config::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << filename << std::endl;
        return false;
    }

    std::string line;
    int lineNumber = 0;
    while (std::getline(file, line)) {
        lineNumber++;
        
        // Remove leading/trailing whitespace
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }
        
        // Parse key=value
        size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) {
            std::cerr << "Warning: Invalid line " << lineNumber << " in " << filename << std::endl;
            continue;
        }
        
        std::string key = trim(line.substr(0, equalPos));
        std::string value = trim(line.substr(equalPos + 1));
        
        if (!key.empty()) {
            settings_[key] = value;
        }
    }
    
    file.close();
    return true;
}

bool Config::parseCommandLine(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.substr(0, 2) == "--") {
            // Long option format: --key=value or --key value
            arg = arg.substr(2);
            size_t equalPos = arg.find('=');
            
            if (equalPos != std::string::npos) {
                std::string key = arg.substr(0, equalPos);
                std::string value = arg.substr(equalPos + 1);
                settings_[key] = value;
            } else {
                // Next argument is the value
                if (i + 1 < argc) {
                    settings_[arg] = argv[i + 1];
                    i++;
                } else {
                    std::cerr << "Error: Missing value for option --" << arg << std::endl;
                    return false;
                }
            }
        } else if (arg[0] == '-' && arg.length() > 1) {
            // Short option format (custom mapping)
            char option = arg[1];
            std::string value;
            
            if (i + 1 < argc) {
                value = argv[i + 1];
                i++;
            } else {
                std::cerr << "Error: Missing value for option -" << option << std::endl;
                return false;
            }
            
            switch (option) {
                case 't':
                    settings_["thread_count"] = value;
                    break;
                case 'i':
                    settings_["input_path"] = value;
                    break;
                case 'o':
                    settings_["output_path"] = value;
                    break;
                case 'b':
                    settings_["buffer_size"] = value;
                    break;
                default:
                    std::cerr << "Warning: Unknown option -" << option << std::endl;
                    break;
            }
        }
    }
    
    return true;
}

std::string Config::getString(const std::string& key) const {
    auto it = settings_.find(key);
    if (it != settings_.end()) {
        return it->second;
    }
    return "";
}

int Config::getInt(const std::string& key) const {
    std::string value = getString(key);
    if (value.empty()) {
        throw std::runtime_error("Configuration key not found: " + key);
    }
    
    try {
        return std::stoi(value);
    } catch (const std::exception& e) {
        throw std::runtime_error("Invalid integer value for key " + key + ": " + value);
    }
}

long Config::getLong(const std::string& key) const {
    std::string value = getString(key);
    if (value.empty()) {
        throw std::runtime_error("Configuration key not found: " + key);
    }
    
    try {
        return std::stol(value);
    } catch (const std::exception& e) {
        throw std::runtime_error("Invalid long value for key " + key + ": " + value);
    }
}

bool Config::getBool(const std::string& key) const {
    std::string value = getString(key);
    if (value.empty()) {
        throw std::runtime_error("Configuration key not found: " + key);
    }
    
    std::string lowerValue = toLower(value);
    return (lowerValue == "true" || lowerValue == "yes" || 
            lowerValue == "1" || lowerValue == "on");
}

void Config::set(const std::string& key, const std::string& value) {
    settings_[key] = value;
}

void Config::set(const std::string& key, int value) {
    settings_[key] = std::to_string(value);
}

void Config::set(const std::string& key, long value) {
    settings_[key] = std::to_string(value);
}

void Config::set(const std::string& key, bool value) {
    settings_[key] = value ? "true" : "false";
}

bool Config::has(const std::string& key) const {
    return settings_.find(key) != settings_.end();
}

void Config::validate() const {
    // Validate thread count
    int threadCount = getInt("thread_count");
    if (threadCount < 1 || threadCount > 128) {
        throw std::runtime_error("Invalid thread_count: must be between 1 and 128");
    }
    
    // Validate buffer sizes
    int bufferSize = getInt("buffer_size");
    if (bufferSize < 512) {
        throw std::runtime_error("Invalid buffer_size: must be at least 512 bytes");
    }
    
    int maxBufferSize = getInt("max_buffer_size");
    if (maxBufferSize < bufferSize) {
        throw std::runtime_error("max_buffer_size must be >= buffer_size");
    }
    
    // Validate paths
    if (getString("input_path").empty()) {
        throw std::runtime_error("input_path cannot be empty");
    }
    
    if (getString("output_path").empty()) {
        throw std::runtime_error("output_path cannot be empty");
    }
    
    // Validate log level
    std::string logLevel = getString("log_level");
    if (logLevel != "debug" && logLevel != "info" && 
        logLevel != "warning" && logLevel != "error") {
        throw std::runtime_error("Invalid log_level: must be debug, info, warning, or error");
    }
}

void Config::printAll() const {
    std::cout << "Current Configuration:" << std::endl;
    std::cout << "=====================" << std::endl;
    
    for (const auto& pair : settings_) {
        std::cout << pair.first << " = " << pair.second << std::endl;
    }
    
    std::cout << "=====================" << std::endl;
}

std::string Config::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

std::string Config::toLower(const std::string& str) const {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

} // namespace dataproc