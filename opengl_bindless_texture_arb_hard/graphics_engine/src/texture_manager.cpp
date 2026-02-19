#include <GL/glew.h>
#include <iostream>
#include <unordered_map>
#include <string>
#include "texture_manager.h"

// TextureManager implementation for handling both traditional and bindless textures
// NOTE: This code contains errors in bindless texture API usage

class TextureManager {
private:
    std::unordered_map<std::string, GLuint> textureHandles;
    std::unordered_map<std::string, GLuint64> bindlessHandles;

public:
    TextureManager() {
        std::cout << "TextureManager initialized" << std::endl;
    }

    ~TextureManager() {
        // Cleanup texture resources
        for (auto& pair : textureHandles) {
            glDeleteTextures(1, &pair.second);
        }
        textureHandles.clear();
        bindlessHandles.clear();
        std::cout << "TextureManager destroyed" << std::endl;
    }

    // Load a texture from file and return the texture ID
    GLuint loadTexture(const std::string& path) {
        GLuint textureID;
        glGenTextures(1, &textureID);
        
        if (textureID == 0) {
            std::cerr << "Failed to generate texture for: " << path << std::endl;
            return 0;
        }

        glBindTexture(GL_TEXTURE_2D, textureID);
        
        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Simulate texture data loading (would normally load from file)
        unsigned char dummyData[16] = {255, 255, 255, 255, 0, 0, 0, 0, 
                                        255, 255, 255, 255, 0, 0, 0, 0};
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, dummyData);
        glGenerateMipmap(GL_TEXTURE_2D);

        glBindTexture(GL_TEXTURE_2D, 0);

        textureHandles[path] = textureID;
        std::cout << "Loaded texture: " << path << " with ID: " << textureID << std::endl;

        return textureID;
    }

    // Get bindless handle for a texture by name
    // ERROR: Uses incorrect API call without ARB suffix
    GLuint64 getBindlessHandle(const std::string& name) {
        auto it = textureHandles.find(name);
        if (it == textureHandles.end()) {
            std::cerr << "Texture not found: " << name << std::endl;
            return 0;
        }

        GLuint textureID = it->second;
        
        // ERROR: Should be glGetTextureHandleARB
        GLuint64 handle = glGetTextureHandle(textureID);
        
        if (handle == 0) {
            std::cerr << "Failed to get bindless handle for: " << name << std::endl;
            return 0;
        }

        bindlessHandles[name] = handle;
        return handle;
    }

    // Create a bindless texture handle from texture ID
    // ERROR: Multiple incorrect API calls
    GLuint64 createBindlessHandle(GLuint textureID) {
        if (textureID == 0) {
            std::cerr << "Invalid texture ID provided" << std::endl;
            return 0;
        }

        // ERROR: Should be glGetTextureHandleARB
        GLuint64 handle = glGetTextureHandle(textureID);
        
        if (handle == 0) {
            std::cerr << "Failed to create bindless handle for texture ID: " << textureID << std::endl;
            return 0;
        }

        // ERROR: Should be glMakeTextureHandleResidentARB
        glMakeTextureHandleResident(handle);

        std::cout << "Created bindless handle: " << handle << " for texture: " << textureID << std::endl;
        return handle;
    }

    // Make a texture handle resident for GPU access
    // ERROR: Missing ARB suffix
    void makeHandleResident(GLuint64 handle) {
        if (handle == 0) {
            std::cerr << "Cannot make resident: invalid handle" << std::endl;
            return;
        }

        // ERROR: Should be glMakeTextureHandleResidentARB
        glMakeTextureHandleResident(handle);
        std::cout << "Made texture handle resident: " << handle << std::endl;
    }

    // Get traditional texture ID by name
    GLuint getTextureID(const std::string& name) {
        auto it = textureHandles.find(name);
        if (it != textureHandles.end()) {
            return it->second;
        }
        return 0;
    }
};