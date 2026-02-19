#include <GL/glew.h>
#include <iostream>
#include <string>
#include "renderer.h"

// Renderer class implementation for OpenGL initialization
// Handles extension checking and basic GL state setup

bool Renderer::initializeGL() {
    std::cout << "Initializing OpenGL renderer..." << std::endl;
    
    // Initialize GLEW to load OpenGL extensions
    GLenum glewStatus = glewInit();
    if (glewStatus != GLEW_OK) {
        std::cerr << "GLEW initialization failed: " 
                  << glewGetErrorString(glewStatus) << std::endl;
        return false;
    }
    
    std::cout << "GLEW initialized successfully" << std::endl;
    
    // Print OpenGL information
    printGLInfo();
    
    // Check for required extensions
    if (!checkExtensions()) {
        std::cerr << "Required extensions not available" << std::endl;
        return false;
    }
    
    // Set up basic OpenGL state
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    
    // Set default clear color
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    
    std::cout << "OpenGL state configured" << std::endl;
    
    return true;
}

// Check for required OpenGL extensions
// Critical: ARB_bindless_texture support is required for modern texture handling
bool Renderer::checkExtensions() {
    std::cout << "Checking for required OpenGL extensions..." << std::endl;
    
    // Check for bindless texture extension
    if (!glewIsSupported("GL_ARB_bindless_texture")) {
        std::cerr << "GL_ARB_bindless_texture is not supported on this hardware" << std::endl;
        m_bindlessTextureSupported = false;
        return false;
    }
    
    std::cout << "GL_ARB_bindless_texture: supported" << std::endl;
    m_bindlessTextureSupported = true;
    
    // Check for other useful extensions
    if (glewIsSupported("GL_ARB_direct_state_access")) {
        std::cout << "GL_ARB_direct_state_access: supported" << std::endl;
        m_dsaSupported = true;
    } else {
        std::cout << "GL_ARB_direct_state_access: not supported" << std::endl;
        m_dsaSupported = false;
    }
    
    return true;
}

// Print OpenGL version and renderer information
void Renderer::printGLInfo() {
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    const GLubyte* glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
    
    std::cout << "OpenGL Renderer: " << renderer << std::endl;
    std::cout << "OpenGL Version: " << version << std::endl;
    std::cout << "GLSL Version: " << glslVersion << std::endl;
}