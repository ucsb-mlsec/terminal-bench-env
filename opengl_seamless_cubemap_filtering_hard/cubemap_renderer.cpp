// cubemap_renderer.cpp
// Graphics Rendering Library - Cubemap Environmental Reflection System
// OpenGL 3.2+ Implementation

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// Global state
GLuint cubemapTexture = 0;
GLuint reflectionShader = 0;
GLuint sphereVAO = 0;
GLuint sphereVBO = 0;
GLuint sphereEBO = 0;
int sphereIndexCount = 0;

// Forward declarations
bool initGL();
void setupCubemap();
void loadCubemapTexture(const std::vector<std::string>& faces);
void createReflectiveSphere();
void renderScene(const glm::mat4& view, const glm::mat4& projection);
GLuint compileShader(const char* source, GLenum type);
GLuint linkShaderProgram(GLuint vertexShader, GLuint fragmentShader);

/**
 * Initialize OpenGL context and set up rendering state
 * Called once at application startup after context creation
 */
bool initGL() {
    // Initialize GLEW to load OpenGL extensions
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW initialization failed: " << glewGetErrorString(err) << std::endl;
        return false;
    }

    // Set viewport dimensions
    glViewport(0, 0, 1280, 720);

    // Enable depth testing for proper 3D rendering
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // Enable face culling for performance
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // NOTE: This is where GL_TEXTURE_CUBE_MAP_SEAMLESS should be enabled
    // to eliminate artifacts at cubemap face boundaries
    
    // Set clear color
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);

    // Print OpenGL version info
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

    return true;
}

/**
 * Set up cubemap texture with proper filtering parameters
 * Configures texture state for environmental reflection mapping
 */
void setupCubemap() {
    std::vector<std::string> faces = {
        "textures/right.jpg",   // GL_TEXTURE_CUBE_MAP_POSITIVE_X
        "textures/left.jpg",    // GL_TEXTURE_CUBE_MAP_NEGATIVE_X
        "textures/top.jpg",     // GL_TEXTURE_CUBE_MAP_POSITIVE_Y
        "textures/bottom.jpg",  // GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
        "textures/front.jpg",   // GL_TEXTURE_CUBE_MAP_POSITIVE_Z
        "textures/back.jpg"     // GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
    };
    
    loadCubemapTexture(faces);
}

/**
 * Load cubemap texture from 6 individual face images
 * Sets up all texture parameters for high-quality filtering
 */
void loadCubemapTexture(const std::vector<std::string>& faces) {
    glGenTextures(1, &cubemapTexture);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);

    // Load each face of the cubemap
    for (unsigned int i = 0; i < faces.size(); i++) {
        // Simulated texture data (in real implementation, load from file)
        int width = 512;
        int height = 512;
        std::vector<unsigned char> data(width * height * 3);
        
        // Fill with procedural gradient for testing
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                data[idx + 0] = (x * 255) / width;
                data[idx + 1] = (y * 255) / height;
                data[idx + 2] = ((i * 40) % 255);
            }
        }

        glTexImage2D(
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
            0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data()
        );
    }

    // Set texture parameters for high-quality filtering
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    // Generate mipmaps for better quality at distance
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

/**
 * Create sphere geometry for reflective object
 * Generates vertex positions, normals, and indices
 */
void createReflectiveSphere() {
    const int latitudeBands = 30;
    const int longitudeBands = 30;
    const float radius = 1.0f;

    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    // Generate sphere vertices
    for (int lat = 0; lat <= latitudeBands; lat++) {
        float theta = lat * glm::pi<float>() / latitudeBands;
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int lon = 0; lon <= longitudeBands; lon++) {
            float phi = lon * 2.0f * glm::pi<float>() / longitudeBands;
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            float x = cosPhi * sinTheta;
            float y = cosTheta;
            float z = sinPhi * sinTheta;

            // Position
            vertices.push_back(radius * x);
            vertices.push_back(radius * y);
            vertices.push_back(radius * z);

            // Normal (same as position for unit sphere)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

    // Generate sphere indices
    for (int lat = 0; lat < latitudeBands; lat++) {
        for (int lon = 0; lon < longitudeBands; lon++) {
            int first = (lat * (longitudeBands + 1)) + lon;
            int second = first + longitudeBands + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }

    sphereIndexCount = indices.size();

    // Create and configure VAO/VBO/EBO
    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO);

    glBindVertexArray(sphereVAO);

    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

/**
 * Compile individual shader from source
 */
GLuint compileShader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation error: " << infoLog << std::endl;
        return 0;
    }

    return shader;
}

/**
 * Link vertex and fragment shaders into program
 */
GLuint linkShaderProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Shader linking error: " << infoLog << std::endl;
        return 0;
    }

    return program;
}

/**
 * Render reflective sphere with cubemap-based environmental reflections
 * Applies view and projection transformations
 */
void renderScene(const glm::mat4& view, const glm::mat4& projection) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Use reflection shader program
    glUseProgram(reflectionShader);

    // Set transformation matrices
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::rotate(model, (float)glfwGetTime() * 0.5f, glm::vec3(0.0f, 1.0f, 0.0f));

    GLint modelLoc = glGetUniformLocation(reflectionShader, "model");
    GLint viewLoc = glGetUniformLocation(reflectionShader, "view");
    GLint projectionLoc = glGetUniformLocation(reflectionShader, "projection");

    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    // Set camera position for reflection calculation
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 5.0f);
    GLint cameraPosLoc = glGetUniformLocation(reflectionShader, "cameraPos");
    glUniform3fv(cameraPosLoc, 1, glm::value_ptr(cameraPos));

    // Bind cubemap texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
    GLint skyboxLoc = glGetUniformLocation(reflectionShader, "skybox");
    glUniform1i(skyboxLoc, 0);

    // Render sphere
    glBindVertexArray(sphereVAO);
    glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}