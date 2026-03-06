#include <GL/glew.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Cubemap face ordering follows OpenGL convention:
// GL_TEXTURE_CUBE_MAP_POSITIVE_X (right)
// GL_TEXTURE_CUBE_MAP_NEGATIVE_X (left)
// GL_TEXTURE_CUBE_MAP_POSITIVE_Y (top)
// GL_TEXTURE_CUBE_MAP_NEGATIVE_Y (bottom)
// GL_TEXTURE_CUBE_MAP_POSITIVE_Z (front)
// GL_TEXTURE_CUBE_MAP_NEGATIVE_Z (back)

GLuint load_cubemap(const char* faces[6]) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrChannels;
    unsigned char* data;

    // Load each face of the cubemap
    for (unsigned int i = 0; i < 6; i++) {
        data = stbi_load(faces[i], &width, &height, &nrChannels, 0);
        
        if (data) {
            GLenum format = GL_RGB;
            if (nrChannels == 1)
                format = GL_RED;
            else if (nrChannels == 3)
                format = GL_RGB;
            else if (nrChannels == 4)
                format = GL_RGBA;

            // Load texture data for the current cubemap face
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0,
                format,
                width,
                height,
                0,
                format,
                GL_UNSIGNED_BYTE,
                data
            );
            
            stbi_image_free(data);
            
            printf("Loaded cubemap face %d: %s (%dx%d)\n", i, faces[i], width, height);
        } else {
            fprintf(stderr, "Failed to load cubemap texture: %s\n", faces[i]);
            stbi_image_free(data);
            return 0;
        }
    }

    // Set up texture parameters for the cubemap
    setup_texture_parameters(textureID);

    return textureID;
}

void setup_texture_parameters(GLuint textureID) {
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
    
    // Configure texture filtering
    // GL_LINEAR provides smooth interpolation between texels
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Configure texture wrapping
    // GL_CLAMP_TO_EDGE prevents sampling artifacts at texture boundaries
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}

GLuint create_cubemap_from_files(const char* directory, const char* extension) {
    const char* faceNames[6] = {
        "right",
        "left",
        "top",
        "bottom",
        "front",
        "back"
    };
    
    char* facePaths[6];
    
    // Construct full file paths for each cubemap face
    for (int i = 0; i < 6; i++) {
        size_t pathLength = strlen(directory) + strlen(faceNames[i]) + strlen(extension) + 3;
        facePaths[i] = (char*)malloc(pathLength);
        
        if (facePaths[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for cubemap path\n");
            // Free previously allocated memory
            for (int j = 0; j < i; j++) {
                free(facePaths[j]);
            }
            return 0;
        }
        
        snprintf(facePaths[i], pathLength, "%s/%s.%s", directory, faceNames[i], extension);
    }
    
    GLuint cubemapID = load_cubemap((const char**)facePaths);
    
    // Clean up allocated memory
    for (int i = 0; i < 6; i++) {
        free(facePaths[i]);
    }
    
    if (cubemapID == 0) {
        fprintf(stderr, "Failed to create cubemap from directory: %s\n", directory);
        return 0;
    }
    
    printf("Successfully created cubemap texture (ID: %u)\n", cubemapID);
    return cubemapID;
}

void delete_cubemap(GLuint textureID) {
    if (textureID != 0) {
        glDeleteTextures(1, &textureID);
        printf("Deleted cubemap texture (ID: %u)\n", textureID);
    }
}

void bind_cubemap(GLuint textureID, GLuint textureUnit) {
    glActiveTexture(GL_TEXTURE0 + textureUnit);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
}