#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <map>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct FramebufferConfig {
    int width;
    int height;
    std::string color_format;
    std::string depth_format;
    std::string stencil_format;
    std::string color_attachment;
    std::string depth_attachment;
    
    FramebufferConfig() : width(800), height(600), 
                          color_format("RGBA8"), 
                          depth_format("DEPTH24"),
                          stencil_format("NONE"),
                          color_attachment("TEXTURE"),
                          depth_attachment("RENDERBUFFER") {}
};

void createDirectory(const char* path) {
    #ifdef _WIN32
        _mkdir(path);
    #else
        mkdir(path, 0755);
    #endif
}

void logToFile(const std::string& filename, const std::string& message) {
    std::ofstream logFile(filename, std::ios::app);
    if (logFile.is_open()) {
        logFile << message << std::endl;
        logFile.close();
    }
}

void clearLogFile(const std::string& filename) {
    std::ofstream logFile(filename, std::ios::trunc);
    logFile.close();
}

FramebufferConfig readFramebufferConfig(const std::string& configPath) {
    FramebufferConfig config;
    std::ifstream configFile(configPath);
    
    if (!configFile.is_open()) {
        std::cerr << "Failed to open config file: " << configPath << std::endl;
        return config;
    }
    
    std::string line;
    while (std::getline(configFile, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "width") config.width = std::stoi(value);
            else if (key == "height") config.height = std::stoi(value);
            else if (key == "color_format") config.color_format = value;
            else if (key == "depth_format") config.depth_format = value;
            else if (key == "stencil_format") config.stencil_format = value;
            else if (key == "color_attachment") config.color_attachment = value;
            else if (key == "depth_attachment") config.depth_attachment = value;
        }
    }
    
    configFile.close();
    return config;
}

GLenum parseColorFormat(const std::string& format) {
    if (format == "RGBA8") return GL_RGBA8;
    if (format == "RGB8") return GL_RGB8;
    if (format == "RGBA16F") return GL_RGBA16F;
    if (format == "RGBA32F") return GL_RGBA32F;
    return GL_RGBA8;
}

GLenum parseDepthFormat(const std::string& format) {
    if (format == "DEPTH24") return GL_DEPTH_COMPONENT24;
    if (format == "DEPTH32") return GL_DEPTH_COMPONENT32;
    if (format == "DEPTH32F") return GL_DEPTH_COMPONENT32F;
    if (format == "DEPTH24_STENCIL8") return GL_DEPTH24_STENCIL8;
    if (format == "DEPTH_COMPONENT") return GL_DEPTH_COMPONENT;
    return GL_DEPTH_COMPONENT24;
}

std::string getFramebufferStatusString(GLenum status) {
    switch (status) {
        case GL_FRAMEBUFFER_COMPLETE:
            return "GL_FRAMEBUFFER_COMPLETE";
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            return "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: Not all framebuffer attachment points are framebuffer attachment complete";
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            return "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: No images are attached to the framebuffer";
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
            return "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: Draw buffer configuration error";
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
            return "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: Read buffer configuration error";
        case GL_FRAMEBUFFER_UNSUPPORTED:
            return "GL_FRAMEBUFFER_UNSUPPORTED: Combination of internal formats violates implementation-dependent restrictions";
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            return "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: Multisample configuration mismatch";
        case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
            return "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS: Layered attachment configuration error";
        default:
            return "UNKNOWN_FRAMEBUFFER_STATUS";
    }
}

bool checkOpenGLError(const std::string& location, const std::string& logPath) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::stringstream ss;
        ss << "OpenGL Error at " << location << ": 0x" << std::hex << err;
        logToFile(logPath, ss.str());
        std::cerr << ss.str() << std::endl;
        return true;
    }
    return false;
}

GLuint createFramebuffer(const FramebufferConfig& config, const std::string& logPath) {
    clearLogFile(logPath);
    logToFile(logPath, "=== Framebuffer Creation Log ===");
    
    std::stringstream configLog;
    configLog << "Configuration: " << config.width << "x" << config.height
              << " Color:" << config.color_format 
              << " Depth:" << config.depth_format
              << " ColorAttach:" << config.color_attachment
              << " DepthAttach:" << config.depth_attachment;
    logToFile(logPath, configLog.str());
    
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    checkOpenGLError("glBindFramebuffer", logPath);
    
    GLuint colorTexture = 0;
    if (config.color_attachment == "TEXTURE") {
        glGenTextures(1, &colorTexture);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        
        GLenum colorFormat = parseColorFormat(config.color_format);
        glTexImage2D(GL_TEXTURE_2D, 0, colorFormat, config.width, config.height, 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                               GL_TEXTURE_2D, colorTexture, 0);
        
        checkOpenGLError("Color texture attachment", logPath);
        logToFile(logPath, "Color attachment: TEXTURE created");
    }
    
    GLuint depthBuffer = 0;
    if (config.depth_attachment == "RENDERBUFFER" && config.depth_format != "NONE") {
        glGenRenderbuffers(1, &depthBuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
        
        GLenum depthFormat = parseDepthFormat(config.depth_format);
        glRenderbufferStorage(GL_RENDERBUFFER, depthFormat, config.width, config.height);
        
        if (config.depth_format == "DEPTH24_STENCIL8") {
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                     GL_RENDERBUFFER, depthBuffer);
        } else {
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                     GL_RENDERBUFFER, depthBuffer);
        }
        
        checkOpenGLError("Depth renderbuffer attachment", logPath);
        logToFile(logPath, "Depth attachment: RENDERBUFFER created");
    } else if (config.depth_attachment == "TEXTURE" && config.depth_format != "NONE") {
        GLuint depthTexture;
        glGenTextures(1, &depthTexture);
        glBindTexture(GL_TEXTURE_2D, depthTexture);
        
        GLenum depthFormat = parseDepthFormat(config.depth_format);
        glTexImage2D(GL_TEXTURE_2D, 0, depthFormat, config.width, config.height,
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D, depthTexture, 0);
        
        checkOpenGLError("Depth texture attachment", logPath);
        logToFile(logPath, "Depth attachment: TEXTURE created");
    }
    
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    std::string statusStr = getFramebufferStatusString(status);
    logToFile(logPath, "Framebuffer Status: " + statusStr);
    
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer incomplete: " << statusStr << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return 0;
    }
    
    logToFile(logPath, "Framebuffer created successfully!");
    return fbo;
}

void renderScene() {
    glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glBegin(GL_TRIANGLES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex2f(0.0f, 0.6f);
        
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex2f(-0.6f, -0.6f);
        
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex2f(0.6f, -0.6f);
    glEnd();
    
    glFlush();
}

bool saveFramebufferToPNG(int width, int height, const std::string& filename) {
    std::vector<unsigned char> pixels(width * height * 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    
    std::vector<unsigned char> flipped(width * height * 4);
    for (int y = 0; y < height; y++) {
        memcpy(&flipped[y * width * 4], 
               &pixels[(height - 1 - y) * width * 4], 
               width * 4);
    }
    
    int result = stbi_write_png(filename.c_str(), width, height, 4, 
                                 flipped.data(), width * 4);
    
    if (result == 0) {
        std::cerr << "Failed to save PNG: " << filename << std::endl;
        return false;
    }
    
    std::cout << "Image saved successfully: " << filename << std::endl;
    return true;
}

int main() {
    createDirectory("../output");
    createDirectory("../logs");
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "Render App", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    
    GLenum glewStatus = glewInit();
    if (glewStatus != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: " 
                  << glewGetErrorString(glewStatus) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    
    FramebufferConfig config = readFramebufferConfig("../config/framebuffer.conf");
    
    std::string logPath = "../logs/framebuffer_status.log";
    GLuint fbo = createFramebuffer(config, logPath);
    
    if (fbo == 0) {
        std::cerr << "Failed to create framebuffer. Check " << logPath << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, config.width, config.height);
    
    renderScene();
    
    bool saveSuccess = saveFramebufferToPNG(config.width, config.height, 
                                            "../output/result.png");
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    if (!saveSuccess) {
        return 1;
    }
    
    std::cout << "Application completed successfully!" << std::endl;
    return 0;
}