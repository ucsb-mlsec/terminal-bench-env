// Debug callback management system for OpenGL
// Handles filtering and logging of OpenGL debug messages based on severity levels

#ifndef CALLBACK_MANAGER_H
#define CALLBACK_MANAGER_H

#include <GL/gl.h>

// Configuration structure for debug message filtering
struct DebugConfig {
    bool logHigh;
    bool logMedium;
    bool logLow;
    bool logNotification;
};

// Load debug configuration from settings
void loadConfig();

// OpenGL debug callback function
// Filters messages based on severity level configuration
void APIENTRY debugCallback(GLenum source, 
                           GLenum type, 
                           GLuint id, 
                           GLenum severity, 
                           GLsizei length, 
                           const GLchar* message, 
                           const void* userParam);

// Initialize the debug callback system
void initializeDebugSystem();

#endif