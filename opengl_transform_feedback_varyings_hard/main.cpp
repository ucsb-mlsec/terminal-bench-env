#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstring>

// Window dimensions
const GLuint WIDTH = 1280;
const GLuint HEIGHT = 720;

// Particle system constants
const int MAX_PARTICLES = 10000;

// Function to read shader source from file
std::string readShaderFile(const char* filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filepath << std::endl;
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Function to compile a shader from file
GLuint compileShader(const char* filepath, GLenum shaderType) {
    std::string sourceStr = readShaderFile(filepath);
    if (sourceStr.empty()) {
        return 0;
    }
    
    const char* source = sourceStr.c_str();
    
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    // Check for compilation errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << filepath << "\n" << infoLog << std::endl;
        return 0;
    }
    
    std::cout << "Successfully compiled shader: " << filepath << std::endl;
    return shader;
}

// Function to link shader program with transform feedback support
GLuint linkProgram(GLuint vertShader, GLuint geomShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    if (geomShader != 0) {
        glAttachShader(program, geomShader);
    }
    
    // Link the program
    glLinkProgram(program);
    
    // Check for linking errors
    GLint success;
    GLchar infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        return 0;
    }
    
    std::cout << "Successfully linked shader program" << std::endl;
    return program;
}

// Function to check OpenGL errors
void checkGLError(const char* stmt) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error " << err << " at " << stmt << std::endl;
    }
}

// Key callback for handling input
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    
    // Create window with OpenGL context
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Transform Feedback Particle System", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    
    // Set viewport dimensions
    glViewport(0, 0, WIDTH, HEIGHT);
    
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    
    // Compile shaders
    std::cout << "\n=== Compiling Shaders ===" << std::endl;
    GLuint vertexShader = compileShader("/home/user/shader_analysis/particle.vert", GL_VERTEX_SHADER);
    GLuint geometryShader = compileShader("/home/user/shader_analysis/particle.geom", GL_GEOMETRY_SHADER);
    
    if (vertexShader == 0 || geometryShader == 0) {
        std::cerr << "Shader compilation failed, exiting..." << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Create shader program
    std::cout << "\n=== Setting up Transform Feedback ===" << std::endl;
    GLuint particleProgram = glCreateProgram();
    glAttachShader(particleProgram, vertexShader);
    glAttachShader(particleProgram, geometryShader);
    
    // ============================================================================
    // TRANSFORM FEEDBACK CONFIGURATION
    // ============================================================================
    // Specify which output variables from the shader should be captured
    // This MUST be called BEFORE linking the program
    // 
    // The varyings array specifies the names of shader output variables that
    // will be written to transform feedback buffers. These must match the
    // 'out' declarations in the vertex or geometry shader.
    //
    // captured_count = 6 variables
    // Mode: GL_INTERLEAVED_ATTRIBS means all variables are written to a single buffer
    //       in the order specified, packed together sequentially
    // ============================================================================
    
    const char* varyings[] = {
        "out_position",      // vec3 - particle world position (3 components)
        "out_velocity",      // vec3 - particle velocity vector (3 components)
        "out_lifetime",      // float - remaining particle lifetime (1 component)
        "terrain_position",  // vec3 - terrain collision point (3 components)
        "terrain_normal",    // vec3 - terrain surface normal (3 components)
        "terrain_height"     // float - terrain height at particle position (1 component)
    };
    
    const GLsizei varyingCount = 6;
    
    std::cout << "Configuring transform feedback varyings:" << std::endl;
    for (int i = 0; i < varyingCount; i++) {
        std::cout << "  [" << i << "] " << varyings[i] << std::endl;
    }
    
    // Configure transform feedback - THIS MUST BE BEFORE LINKING
    glTransformFeedbackVaryings(particleProgram, varyingCount, varyings, GL_INTERLEAVED_ATTRIBS);
    checkGLError("glTransformFeedbackVaryings");
    
    std::cout << "Transform feedback mode: GL_INTERLEAVED_ATTRIBS" << std::endl;
    std::cout << "Total captured variables: " << varyingCount << std::endl;
    
    // Now link the program with transform feedback configuration
    std::cout << "\n=== Linking Shader Program ===" << std::endl;
    GLuint linkedProgram = linkProgram(vertexShader, geometryShader);
    
    if (linkedProgram == 0) {
        std::cerr << "Program linking failed, exiting..." << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Verify transform feedback setup
    GLint tfVaryingCount;
    glGetProgramiv(linkedProgram, GL_TRANSFORM_FEEDBACK_VARYINGS, &tfVaryingCount);
    std::cout << "Verified transform feedback varyings: " << tfVaryingCount << std::endl;
    
    GLint tfBufferMode;
    glGetProgramiv(linkedProgram, GL_TRANSFORM_FEEDBACK_BUFFER_MODE, &tfBufferMode);
    std::cout << "Buffer mode: " << (tfBufferMode == GL_INTERLEAVED_ATTRIBS ? "INTERLEAVED" : "SEPARATE") << std::endl;
    
    // Query each varying's properties
    std::cout << "\nTransform feedback varying details:" << std::endl;
    for (int i = 0; i < tfVaryingCount; i++) {
        char name[256];
        GLsizei length;
        GLsizei size;
        GLenum type;
        glGetTransformFeedbackVarying(linkedProgram, i, sizeof(name), &length, &size, &type, name);
        std::cout << "  [" << i << "] " << name << " (type=" << type << ", size=" << size << ")" << std::endl;
    }
    
    // Clean up shader objects after linking
    glDeleteShader(vertexShader);
    glDeleteShader(geometryShader);
    
    // Setup vertex data and buffers
    std::cout << "\n=== Setting up Vertex Buffers ===" << std::endl;
    
    // Create VAO and VBOs for particle data
    GLuint VAO, VBO[2];
    glGenVertexArrays(1, &VAO);
    glGenBuffers(2, VBO);
    
    glBindVertexArray(VAO);
    
    // Initial particle data (position, velocity, lifetime)
    std::vector<float> particleData;
    for (int i = 0; i < MAX_PARTICLES; i++) {
        // Position (vec3)
        particleData.push_back(((float)rand() / RAND_MAX) * 20.0f - 10.0f);
        particleData.push_back(((float)rand() / RAND_MAX) * 5.0f);
        particleData.push_back(((float)rand() / RAND_MAX) * 20.0f - 10.0f);
        
        // Velocity (vec3)
        particleData.push_back(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
        particleData.push_back(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
        particleData.push_back(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
        
        // Lifetime (float)
        particleData.push_back(((float)rand() / RAND_MAX) * 5.0f + 1.0f);
    }
    
    // Setup VBO for source data
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, particleData.size() * sizeof(float), particleData.data(), GL_DYNAMIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Velocity attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Lifetime attribute
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    
    // Create transform feedback buffer objects
    // For interleaved mode, we need one buffer to store all varying outputs
    std::cout << "\n=== Setting up Transform Feedback Buffers ===" << std::endl;
    
    GLuint transformFeedbackBuffer;
    glGenBuffers(1, &transformFeedbackBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, transformFeedbackBuffer);
    
    // Calculate size needed for transform feedback buffer
    // out_position (vec3) + out_velocity (vec3) + out_lifetime (float) + 
    // terrain_position (vec3) + terrain_normal (vec3) + terrain_height (float) = 14 floats per particle
    size_t tfBufferSize = MAX_PARTICLES * 14 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, tfBufferSize, nullptr, GL_DYNAMIC_COPY);
    
    std::cout << "Transform feedback buffer size: " << tfBufferSize << " bytes" << std::endl;
    std::cout << "Components per particle: 14 (3+3+1+3+3+1)" << std::endl;
    
    // Create transform feedback object
    GLuint transformFeedback;
    glGenTransformFeedbacks(1, &transformFeedback);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, transformFeedback);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, transformFeedbackBuffer);
    
    checkGLError("Transform feedback buffer setup");
    
    // Unbind
    glBindVertexArray(0);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    
    // Enable some OpenGL features
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize(2.0f);
    
    std::cout << "\n=== Starting Render Loop ===" << std::endl;
    
    // Render loop
    double lastTime = glfwGetTime();
    int frameCount = 0;
    
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastTime >= 1.0) {
            std::cout << "FPS: " << frameCount << " | Particles: " << MAX_PARTICLES << std::endl;
            frameCount = 0;
            lastTime = currentTime;
        }
        
        // Poll events
        glfwPollEvents();
        
        // Clear buffers
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Use shader program
        glUseProgram(linkedProgram);
        
        // Set uniforms (time, etc.)
        GLint timeLocation = glGetUniformLocation(linkedProgram, "u_time");
        glUniform1f(timeLocation, (float)currentTime);
        
        GLint deltaTimeLocation = glGetUniformLocation(linkedProgram, "u_deltaTime");
        glUniform1f(deltaTimeLocation, 0.016f); // Assuming 60 FPS
        
        // Bind VAO and transform feedback
        glBindVertexArray(VAO);
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, transformFeedback);
        
        // Begin transform feedback
        // GL_POINTS because we're processing particles
        glEnable(GL_RASTERIZER_DISCARD); // Disable rasterization for pure compute pass
        glBeginTransformFeedback(GL_POINTS);
        
        // Draw particles (this triggers the vertex/geometry shader and captures output)
        glDrawArrays(GL_POINTS, 0, MAX_PARTICLES);
        
        // End transform feedback
        glEndTransformFeedback();
        glDisable(GL_RASTERIZER_DISCARD);
        
        // Optionally, you could read back the transform feedback buffer here
        // and use it as input for the next frame (ping-pong buffers)
        
        glBindVertexArray(0);
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
        
        // Swap buffers
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    std::cout << "\n=== Cleaning up resources ===" << std::endl;
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(2, VBO);
    glDeleteBuffers(1, &transformFeedbackBuffer);
    glDeleteTransformFeedbacks(1, &transformFeedback);
    glDeleteProgram(linkedProgram);
    
    glfwTerminate();
    return 0;
}