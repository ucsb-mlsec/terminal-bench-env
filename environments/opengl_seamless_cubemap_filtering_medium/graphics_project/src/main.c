#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Forward declarations from other modules
extern int init_renderer(GLFWwindow* window);
extern int setup_textures(void);
extern void render_loop(GLFWwindow* window);
extern void cleanup(void);

// Error callback for GLFW
void error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Window resize callback
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main(int argc, char** argv) {
    GLFWwindow* window = NULL;
    
    printf("Starting graphics application...\n");
    
    // Set error callback before GLFW initialization
    glfwSetErrorCallback(error_callback);
    
    // Initialize GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return EXIT_FAILURE;
    }
    
    // Configure GLFW window hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4); // 4x MSAA
    
    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif
    
    // Create window
    window = glfwCreateWindow(1280, 720, "Graphics Project - Cubemap Renderer", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    
    // Make OpenGL context current
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    // Enable VSync
    glfwSwapInterval(1);
    
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(glew_status));
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }
    
    printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    printf("GLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf("Renderer: %s\n", glGetString(GL_RENDERER));
    
    // Initialize renderer module
    // This sets up shaders, framebuffers, and OpenGL state
    if (init_renderer(window) != 0) {
        fprintf(stderr, "Failed to initialize renderer\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }
    
    // Setup textures including cubemaps
    // This loads and configures all texture resources
    if (setup_textures() != 0) {
        fprintf(stderr, "Failed to setup textures\n");
        cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }
    
    printf("Initialization complete. Starting render loop...\n");
    
    // Enter the main render loop
    render_loop(window);
    
    // Cleanup resources
    printf("Shutting down...\n");
    cleanup();
    
    // Destroy window and terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();
    
    printf("Application terminated successfully.\n");
    return EXIT_SUCCESS;
}