#include <iostream>
#include <string>
#include <GL/gl.h>
#include <fstream>

// Global configuration structure to store debug logging preferences
struct DebugConfig {
    bool logHigh;
    bool logMedium;
    bool logLow;
    bool logNotification;
} config;

// Function to load configuration from file
// Reads debug_config.ini and sets the appropriate flags
void loadConfig() {
    std::ifstream configFile("config/debug_config.ini");
    std::string line;
    
    // Default values
    config.logHigh = true;
    config.logMedium = true;
    config.logLow = false;
    config.logNotification = false;
    
    if (configFile.is_open()) {
        while (std::getline(configFile, line)) {
            if (line.find("logHigh=true") != std::string::npos) {
                config.logHigh = true;
            } else if (line.find("logHigh=false") != std::string::npos) {
                config.logHigh = false;
            } else if (line.find("logMedium=true") != std::string::npos) {
                config.logMedium = true;
            } else if (line.find("logMedium=false") != std::string::npos) {
                config.logMedium = false;
            } else if (line.find("logLow=true") != std::string::npos) {
                config.logLow = true;
            } else if (line.find("logLow=false") != std::string::npos) {
                config.logLow = false;
            } else if (line.find("logNotification=true") != std::string::npos) {
                config.logNotification = true;
            } else if (line.find("logNotification=false") != std::string::npos) {
                config.logNotification = false;
            }
        }
        configFile.close();
    }
}

// OpenGL debug callback function
// This function is called by OpenGL when debug events occur
void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, 
                           GLenum severity, GLsizei length, 
                           const GLchar* message, const void* userParam) {
    
    // Filter messages based on severity level and configuration
    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:
            // Check if high severity messages should be logged
            if (config.logHigh) {
                std::cout << "[HIGH] " << message << std::endl;
            }
            break;
            
        case GL_DEBUG_SEVERITY_MEDIUM:
            // Check if medium severity messages should be logged
            if (config.logMedium) {
                std::cout << "[MEDIUM] " << message << std::endl;
            }
            break;
            
        case GL_DEBUG_SEVERITY_LOW:
            // Check if low severity messages should be logged
            // BUG: This incorrectly checks logNotification instead of logLow
            if (config.logNotification) {
                std::cout << "[LOW] " << message << std::endl;
            }
            break;
            
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            // Check if notification messages should be logged
            // BUG: This incorrectly checks logLow instead of logNotification
            if (config.logLow) {
                std::cout << "[NOTIFICATION] " << message << std::endl;
            }
            break;
            
        default:
            // Handle unknown severity levels
            std::cout << "[UNKNOWN] " << message << std::endl;
            break;
    }
}

// Main function
int main() {
    // Load the debug configuration from file
    loadConfig();
    
    std::cout << "OpenGL Debug Callback System initialized" << std::endl;
    std::cout << "Configuration loaded:" << std::endl;
    std::cout << "  High: " << (config.logHigh ? "enabled" : "disabled") << std::endl;
    std::cout << "  Medium: " << (config.logMedium ? "enabled" : "disabled") << std::endl;
    std::cout << "  Low: " << (config.logLow ? "enabled" : "disabled") << std::endl;
    std::cout << "  Notification: " << (config.logNotification ? "enabled" : "disabled") << std::endl;
    
    // In a real application, this is where we would:
    // 1. Create an OpenGL context with debug support
    // 2. Register the debug callback with glDebugMessageCallback()
    // 3. Enable debug output with glEnable(GL_DEBUG_OUTPUT)
    
    return 0;
}