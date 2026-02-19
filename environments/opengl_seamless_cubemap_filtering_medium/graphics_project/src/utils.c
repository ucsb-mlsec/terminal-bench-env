#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>

#define DEBUG 1

/**
 * Reads an entire text file into a dynamically allocated string buffer
 * Returns NULL on failure
 */
char* read_file(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "Error: Failed to allocate memory for file %s\n", filename);
        fclose(file);
        return NULL;
    }

    size_t bytes_read = fread(buffer, 1, file_size, file);
    buffer[bytes_read] = '\0';

    fclose(file);
    return buffer;
}

/**
 * Converts OpenGL error codes to human-readable strings
 */
const char* gl_error_to_string(GLenum error) {
    switch (error) {
        case GL_NO_ERROR:
            return "GL_NO_ERROR";
        case GL_INVALID_ENUM:
            return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE:
            return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION:
            return "GL_INVALID_OPERATION";
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case GL_OUT_OF_MEMORY:
            return "GL_OUT_OF_MEMORY";
        case GL_STACK_UNDERFLOW:
            return "GL_STACK_UNDERFLOW";
        case GL_STACK_OVERFLOW:
            return "GL_STACK_OVERFLOW";
        default:
            return "UNKNOWN_ERROR";
    }
}

/**
 * Checks for OpenGL errors and prints error messages
 * Returns 1 if error found, 0 otherwise
 */
int check_gl_error(const char* context) {
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL Error in %s: %s (0x%x)\n", 
                context, gl_error_to_string(error), error);
        return 1;
    }
    return 0;
}

/**
 * Conditionally prints debug messages based on DEBUG flag
 */
void print_debug(const char* format, ...) {
    if (DEBUG) {
        va_list args;
        va_start(args, format);
        printf("[DEBUG] ");
        vprintf(format, args);
        printf("\n");
        va_end(args);
    }
}

/**
 * Checks if a file exists
 * Returns 1 if file exists, 0 otherwise
 */
int file_exists(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file) {
        fclose(file);
        return 1;
    }
    return 0;
}