#include "logger.h"
#include <iostream>
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace loglib {

// Helper function to get current timestamp
static std::string get_timestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Helper function to format log message
static std::string format_message(const std::string& level, const std::string& message) {
    std::ostringstream oss;
    oss << "[" << get_timestamp() << "] [" << level << "] " << message;
    return oss.str();
}

void log_info(const char* message) {
    if (message) {
        std::cout << format_message("INFO", message) << std::endl;
    }
}

void log_info(const std::string& message) {
    std::cout << format_message("INFO", message) << std::endl;
}

void log_warning(const char* message) {
    if (message) {
        std::cout << format_message("WARNING", message) << std::endl;
    }
}

void log_warning(const std::string& message) {
    std::cout << format_message("WARNING", message) << std::endl;
}

void log_error(const char* message) {
    if (message) {
        std::cerr << format_message("ERROR", message) << std::endl;
    }
}

void log_error(const std::string& message) {
    std::cerr << format_message("ERROR", message) << std::endl;
}

void log_debug(const char* message) {
    if (message) {
        std::cout << format_message("DEBUG", message) << std::endl;
    }
}

void log_debug(const std::string& message) {
    std::cout << format_message("DEBUG", message) << std::endl;
}

Logger::Logger(const std::string& name) : logger_name(name) {}

void Logger::info(const std::string& message) {
    std::cout << format_message("INFO", "[" + logger_name + "] " + message) << std::endl;
}

void Logger::warning(const std::string& message) {
    std::cout << format_message("WARNING", "[" + logger_name + "] " + message) << std::endl;
}

void Logger::error(const std::string& message) {
    std::cerr << format_message("ERROR", "[" + logger_name + "] " + message) << std::endl;
}

void Logger::debug(const std::string& message) {
    std::cout << format_message("DEBUG", "[" + logger_name + "] " + message) << std::endl;
}

} // namespace loglib