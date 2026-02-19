#include <GL/glew.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Compile a shader from source code
GLuint compile_shader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    if (shader == 0) {
        fprintf(stderr, "Error: Failed to create shader\n");
        return 0;
    }

    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    // Check compilation status
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint info_len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_len);
        
        if (info_len > 1) {
            char* info_log = (char*)malloc(info_len);
            glGetShaderInfoLog(shader, info_len, NULL, info_log);
            fprintf(stderr, "Error compiling shader:\n%s\n", info_log);
            free(info_log);
        }
        
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

// Link vertex and fragment shaders into a program
GLuint link_program(GLuint vertex_shader, GLuint fragment_shader) {
    GLuint program = glCreateProgram();
    if (program == 0) {
        fprintf(stderr, "Error: Failed to create shader program\n");
        return 0;
    }

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    // Check link status
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint info_len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_len);
        
        if (info_len > 1) {
            char* info_log = (char*)malloc(info_len);
            glGetProgramInfoLog(program, info_len, NULL, info_log);
            fprintf(stderr, "Error linking program:\n%s\n", info_log);
            free(info_log);
        }
        
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

// Load shader source code from a file
char* load_shader_from_file(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open shader file: %s\n", filename);
        return NULL;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate buffer and read file
    char* buffer = (char*)malloc(size + 1);
    if (!buffer) {
        fprintf(stderr, "Error: Failed to allocate memory for shader source\n");
        fclose(file);
        return NULL;
    }

    size_t bytes_read = fread(buffer, 1, size, file);
    buffer[bytes_read] = '\0';
    fclose(file);

    return buffer;
}

// Create a complete shader program from file paths
GLuint create_program_from_files(const char* vertex_path, const char* fragment_path) {
    char* vertex_source = load_shader_from_file(vertex_path);
    if (!vertex_source) {
        return 0;
    }

    char* fragment_source = load_shader_from_file(fragment_path);
    if (!fragment_source) {
        free(vertex_source);
        return 0;
    }

    GLuint vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER);
    GLuint fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER);

    free(vertex_source);
    free(fragment_source);

    if (vertex_shader == 0 || fragment_shader == 0) {
        if (vertex_shader != 0) glDeleteShader(vertex_shader);
        if (fragment_shader != 0) glDeleteShader(fragment_shader);
        return 0;
    }

    GLuint program = link_program(vertex_shader, fragment_shader);

    // Clean up shaders (they're now in the program)
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return program;
}