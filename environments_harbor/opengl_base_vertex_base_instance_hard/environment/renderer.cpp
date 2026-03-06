#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <iostream>
#include <fstream>

// Vertex structure with position, normal, and color data
struct Vertex {
    float position[3];  // x, y, z
    float normal[3];    // nx, ny, nz
    float color[4];     // r, g, b, a
};

// Global OpenGL objects
GLuint shaderProgram, VAO, VBO, EBO;

// Mesh vertex counts - each mesh has a specific number of vertices
const int CUBE_VERTEX_COUNT = 24;      // 6 faces * 4 vertices per face
const int PYRAMID_VERTEX_COUNT = 18;   // 5 faces with varying vertex counts
const int SPHERE_VERTEX_COUNT = 32;    // Approximation with limited vertices

// Mesh index counts - number of indices needed to render each mesh
const int CUBE_INDEX_COUNT = 36;       // 6 faces * 2 triangles * 3 indices
const int PYRAMID_INDEX_COUNT = 18;    // 6 triangles * 3 indices
const int SPHERE_INDEX_COUNT = 60;     // 20 triangles * 3 indices

// Number of instances to render for each mesh type
const int CUBE_INSTANCES = 3;
const int PYRAMID_INSTANCES = 2;
const int SPHERE_INSTANCES = 4;

// Function to create cube vertex data (24 vertices - 4 per face for proper normals)
std::vector<Vertex> createCubeVertices() {
    std::vector<Vertex> vertices;
    
    // Front face (red)
    vertices.push_back({{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}});
    vertices.push_back({{0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}});
    vertices.push_back({{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}});
    vertices.push_back({{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}});
    
    // Back face (green)
    vertices.push_back({{0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
    vertices.push_back({{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
    vertices.push_back({{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
    vertices.push_back({{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
    
    // Top face (blue)
    vertices.push_back({{-0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
    vertices.push_back({{0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
    vertices.push_back({{0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
    vertices.push_back({{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
    
    // Bottom face (yellow)
    vertices.push_back({{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}});
    vertices.push_back({{0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}});
    vertices.push_back({{0.5f, -0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}});
    vertices.push_back({{-0.5f, -0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}});
    
    // Right face (magenta)
    vertices.push_back({{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 1.0f}});
    vertices.push_back({{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 1.0f}});
    vertices.push_back({{0.5f, 0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 1.0f}});
    vertices.push_back({{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 1.0f}});
    
    // Left face (cyan)
    vertices.push_back({{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 1.0f}});
    vertices.push_back({{-0.5f, -0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 1.0f}});
    vertices.push_back({{-0.5f, 0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 1.0f}});
    vertices.push_back({{-0.5f, 0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 1.0f}});
    
    return vertices;
}

// Function to create cube index data
std::vector<unsigned int> createCubeIndices() {
    std::vector<unsigned int> indices;
    
    // Each face is defined by 2 triangles (6 indices)
    for (int face = 0; face < 6; face++) {
        int base = face * 4;
        indices.push_back(base + 0);
        indices.push_back(base + 1);
        indices.push_back(base + 2);
        indices.push_back(base + 0);
        indices.push_back(base + 2);
        indices.push_back(base + 3);
    }
    
    return indices;
}

// Function to create pyramid vertex data (18 vertices - 3 per triangle for proper normals)
std::vector<Vertex> createPyramidVertices() {
    std::vector<Vertex> vertices;
    
    // Base square (2 triangles)
    vertices.push_back({{-0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.8f, 0.2f, 0.2f, 1.0f}});
    vertices.push_back({{0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.8f, 0.2f, 0.2f, 1.0f}});
    vertices.push_back({{0.5f, 0.0f, 0.5f}, {0.0f, -1.0f, 0.0f}, {0.8f, 0.2f, 0.2f, 1.0f}});
    
    vertices.push_back({{-0.5f, 0.0f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.8f, 0.2f, 0.2f, 1.0f}});
    vertices.push_back({{0.5f, 0.0f, 0.5f}, {0.0f, -1.0f, 0.0f}, {0.8f, 0.2f, 0.2f, 1.0f}});
    vertices.push_back({{-0.5f, 0.0f, 0.5f}, {0.0f, -1.0f, 0.0f}, {0.8f, 0.2f, 0.2f, 1.0f}});
    
    // Front face
    vertices.push_back({{-0.5f, 0.0f, 0.5f}, {0.0f, 0.447f, 0.894f}, {0.2f, 0.8f, 0.2f, 1.0f}});
    vertices.push_back({{0.5f, 0.0f, 0.5f}, {0.0f, 0.447f, 0.894f}, {0.2f, 0.8f, 0.2f, 1.0f}});
    vertices.push_back({{0.0f, 1.0f, 0.0f}, {0.0f, 0.447f, 0.894f}, {0.2f, 0.8f, 0.2f, 1.0f}});
    
    // Right face
    vertices.push_back({{0.5f, 0.0f, 0.5f}, {0.894f, 0.447f, 0.0f}, {0.2f, 0.2f, 0.8f, 1.0f}});
    vertices.push_back({{0.5f, 0.0f, -0.5f}, {0.894f, 0.447f, 0.0f}, {0.2f, 0.2f, 0.8f, 1.0f}});
    vertices.push_back({{0.0f, 1.0f, 0.0f}, {0.894f, 0.447f, 0.0f}, {0.2f, 0.2f, 0.8f, 1.0f}});
    
    // Back face
    vertices.push_back({{0.5f, 0.0f, -0.5f}, {0.0f, 0.447f, -0.894f}, {0.8f, 0.8f, 0.2f, 1.0f}});
    vertices.push_back({{-0.5f, 0.0f, -0.5f}, {0.0f, 0.447f, -0.894f}, {0.8f, 0.8f, 0.2f, 1.0f}});
    vertices.push_back({{0.0f, 1.0f, 0.0f}, {0.0f, 0.447f, -0.894f}, {0.8f, 0.8f, 0.2f, 1.0f}});
    
    // Left face
    vertices.push_back({{-0.5f, 0.0f, -0.5f}, {-0.894f, 0.447f, 0.0f}, {0.8f, 0.2f, 0.8f, 1.0f}});
    vertices.push_back({{-0.5f, 0.0f, 0.5f}, {-0.894f, 0.447f, 0.0f}, {0.8f, 0.2f, 0.8f, 1.0f}});
    vertices.push_back({{0.0f, 1.0f, 0.0f}, {-0.894f, 0.447f, 0.0f}, {0.8f, 0.2f, 0.8f, 1.0f}});
    
    return vertices;
}

// Function to create pyramid index data
std::vector<unsigned int> createPyramidIndices() {
    std::vector<unsigned int> indices;
    
    // Simply sequential indices for 6 triangles
    for (unsigned int i = 0; i < 18; i++) {
        indices.push_back(i);
    }
    
    return indices;
}

// Function to create sphere approximation vertex data (32 vertices)
std::vector<Vertex> createSphereVertices() {
    std::vector<Vertex> vertices;
    float radius = 0.5f;
    
    // Create a simple icosahedron-like structure with 32 vertices
    // Top vertex
    vertices.push_back({{0.0f, radius, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.9f, 0.3f, 0.3f, 1.0f}});
    
    // Upper ring (8 vertices)
    for (int i = 0; i < 8; i++) {
        float angle = i * 3.14159f * 2.0f / 8.0f;
        float x = radius * 0.7f * cos(angle);
        float y = radius * 0.5f;
        float z = radius * 0.7f * sin(angle);
        glm::vec3 normal = glm::normalize(glm::vec3(x, y, z));
        vertices.push_back({{x, y, z}, {normal.x, normal.y, normal.z}, {0.3f, 0.9f, 0.3f, 1.0f}});
    }
    
    // Middle ring (8 vertices)
    for (int i = 0; i < 8; i++) {
        float angle = (i + 0.5f) * 3.14159f * 2.0f / 8.0f;
        float x = radius * cos(angle);
        float y = 0.0f;
        float z = radius * sin(angle);
        glm::vec3 normal = glm::normalize(glm::vec3(x, y, z));
        vertices.push_back({{x, y, z}, {normal.x, normal.y, normal.z}, {0.3f, 0.3f, 0.9f, 1.0f}});
    }
    
    // Lower ring (8 vertices)
    for (int i = 0; i < 8; i++) {
        float angle = i * 3.14159f * 2.0f / 8.0f;
        float x = radius * 0.7f * cos(angle);
        float y = -radius * 0.5f;
        float z = radius * 0.7f * sin(angle);
        glm::vec3 normal = glm::normalize(glm::vec3(x, y, z));
        vertices.push_back({{x, y, z}, {normal.x, normal.y, normal.z}, {0.9f, 0.9f, 0.3f, 1.0f}});
    }
    
    // Bottom vertex
    vertices.push_back({{0.0f, -radius, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.9f, 0.3f, 0.9f, 1.0f}});
    
    // Fill remaining vertices to reach 32
    for (int i = 0; i < 6; i++) {
        float angle = i * 3.14159f * 2.0f / 6.0f;
        float x = radius * 0.5f * cos(angle);
        float y = radius * 0.25f;
        float z = radius * 0.5f * sin(angle);
        glm::vec3 normal = glm::normalize(glm::vec3(x, y, z));
        vertices.push_back({{x, y, z}, {normal.x, normal.y, normal.z}, {0.3f, 0.9f, 0.9f, 1.0f}});
    }
    
    return vertices;
}

// Function to create sphere index data (60 indices = 20 triangles)
std::vector<unsigned int> createSphereIndices() {
    std::vector<unsigned int> indices;
    
    // Top cap triangles (8 triangles)
    for (int i = 0; i < 8; i++) {
        indices.push_back(0);  // Top vertex
        indices.push_back(1 + i);
        indices.push_back(1 + (i + 1) % 8);
    }
    
    // Middle band triangles (8 triangles)
    for (int i = 0; i < 8; i++) {
        indices.push_back(1 + i);
        indices.push_back(9 + i);
        indices.push_back(1 + (i + 1) % 8);
    }
    
    // Bottom cap triangles (4 triangles)
    for (int i = 0; i < 4; i++) {
        indices.push_back(17 + i);
        indices.push_back(25);  // Bottom vertex
        indices.push_back(17 + (i + 1) % 8);
    }
    
    return indices;
}

// Setup function to create and upload all buffer data
void setupBuffers() {
    // Create vertex data for all three mesh types
    std::vector<Vertex> cubeVertices = createCubeVertices();
    std::vector<Vertex> pyramidVertices = createPyramidVertices();
    std::vector<Vertex> sphereVertices = createSphereVertices();
    
    // Create index data for all three mesh types
    std::vector<unsigned int> cubeIndices = createCubeIndices();
    std::vector<unsigned int> pyramidIndices = createPyramidIndices();
    std::vector<unsigned int> sphereIndices = createSphereIndices();
    
    // Combine all vertex data into a single buffer
    // Layout: [cube vertices][pyramid vertices][sphere vertices]
    std::vector<Vertex> allVertices;
    allVertices.insert(allVertices.end(), cubeVertices.begin(), cubeVertices.end());
    allVertices.insert(allVertices.end(), pyramidVertices.begin(), pyramidVertices.end());
    allVertices.insert(allVertices.end(), sphereVertices.begin(), sphereVertices.end());
    
    // Combine all index data into a single buffer
    // Layout: [cube indices][pyramid indices][sphere indices]
    std::vector<unsigned int> allIndices;
    allIndices.insert(allIndices.end(), cubeIndices.begin(), cubeIndices.end());
    allIndices.insert(allIndices.end(), pyramidIndices.begin(), pyramidIndices.end());
    allIndices.insert(allIndices.end(), sphereIndices.begin(), sphereIndices.end());
    
    std::cout << "Buffer Layout Information:" << std::endl;
    std::cout << "Total vertices: " << allVertices.size() << std::endl;
    std::cout << "Total indices: " << allIndices.size() << std::endl;
    std::cout << "Cube: vertices 0-" << (CUBE_VERTEX_COUNT-1) << ", indices 0-" << (CUBE_INDEX_COUNT-1) << std::endl;
    std::cout << "Pyramid: vertices " << CUBE_VERTEX_COUNT << "-" << (CUBE_VERTEX_COUNT+PYRAMID_VERTEX_COUNT-1) 
              << ", indices " << CUBE_INDEX_COUNT << "-" << (CUBE_INDEX_COUNT+PYRAMID_INDEX_COUNT-1) << std::endl;
    std::cout << "Sphere: vertices " << (CUBE_VERTEX_COUNT+PYRAMID_VERTEX_COUNT) << "-" 
              << (CUBE_VERTEX_COUNT+PYRAMID_VERTEX_COUNT+SPHERE_VERTEX_COUNT-1)
              << ", indices " << (CUBE_INDEX_COUNT+PYRAMID_INDEX_COUNT) << "-" 
              << (CUBE_INDEX_COUNT+PYRAMID_INDEX_COUNT+SPHERE_INDEX_COUNT-1) << std::endl;
    
    // Create and bind Vertex Array Object
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    
    // Create and bind Vertex Buffer Object
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, allVertices.size() * sizeof(Vertex), allVertices.data(), GL_STATIC_DRAW);
    
    // Create and bind Element Buffer Object (index buffer)
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, allIndices.size() * sizeof(unsigned int), allIndices.data(), GL_STATIC_DRAW);
    
    // Set up vertex attribute pointers
    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
    glEnableVertexAttribArray(0);
    
    // Normal attribute (location 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(1);
    
    // Color attribute (location 2)
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, color));
    glEnableVertexAttribArray(2);
    
    // Unbind VAO
    glBindVertexArray(0);
}

// Render scene function with INTENTIONALLY BROKEN offset calculations
// The base vertex and base index values are incorrect, causing vertex corruption
void renderScene() {
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    
    // Get uniform locations
    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    
    // Set up view and projection matrices
    glm::mat4 view = glm::lookAt(glm::vec3(5.0f, 5.0f, 10.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
    
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projection[0][0]);
    
    // TODO: Fix these offset calculations!
    // The base vertex and base index values are incorrect, causing vertex corruption
    
    // Render cube instances with INCORRECT offsets
    // BUG: These values don't match the actual buffer layout!
    int cube_base_vertex = 5;      // WRONG! Should be 0
    int cube_base_index = 10;      // WRONG! Should be 0
    
    std::cout << "Rendering cubes with base_vertex=" << cube_base_vertex 
              << ", base_index=" << cube_base_index << std::endl;
    
    for (int i = 0; i < CUBE_INSTANCES; i++) {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(-4.0f + i * 2.0f, 0.0f, 0.0f));
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model[0][0]);
        
        // Using glDrawElementsBaseVertex with incorrect base vertex offset
        glDrawElementsBaseVertex(
            GL_TRIANGLES, 
            CUBE_INDEX_COUNT, 
            GL_UNSIGNED_INT, 
            (void*)(cube_base_index * sizeof(unsigned int)),
            cube_base_vertex
        );
    }
    
    // Render pyramid instances with INCORRECT offsets
    // BUG: These values don't match the actual buffer layout!
    int pyramid_base_vertex = 20;  // WRONG! Should be 24 (after cube vertices)
    int pyramid_base_index = 30;   // WRONG! Should be 36 (after cube indices)
    
    std::cout << "Rendering pyramids with base_vertex=" << pyramid_base_vertex 
              << ", base_index=" << pyramid_base_index << std::endl;
    
    for (int i = 0; i < PYRAMID_INSTANCES; i++) {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(-2.0f + i * 4.0f, 2.0f, -2.0f));
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model[0][0]);
        
        // Using glDrawElementsBaseVertex with incorrect base vertex offset
        glDrawElementsBaseVertex(
            GL_TRIANGLES, 
            PYRAMID_INDEX_COUNT, 
            GL_UNSIGNED_INT, 
            (void*)(pyramid_base_index * sizeof(unsigned int)),
            pyramid_base_vertex
        );
    }
    
    // Render sphere instances with INCORRECT offsets
    // BUG: These values don't match the actual buffer layout!
    int sphere_base_vertex = 40;   // WRONG! Should be 42 (after cube and pyramid vertices)
    int sphere_base_index = 70;    // WRONG! Should be 54 (after cube and pyramid indices)
    
    std::cout << "Rendering spheres with base_vertex=" << sphere_base_vertex 
              << ", base_index=" << sphere_base_index << std::endl;
    
    for (int i = 0; i < SPHERE_INSTANCES; i++) {
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(-3.0f + i * 2.0f, -2.0f, 2.0f));
        model = glm::scale(model, glm::vec3(0.8f));
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model[0][0]);
        
        // Using glDrawElementsBaseVertex with incorrect base vertex offset
        glDrawElementsBaseVertex(
            GL_TRIANGLES, 
            SPHERE_INDEX_COUNT, 
            GL_UNSIGNED_INT, 
            (void*)(sphere_base_index * sizeof(unsigned int)),
            sphere_base_vertex
        );
    }
    
    glBindVertexArray(0);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 600, "Broken Mesh Rendering", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    
    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    
    // Setup buffers with combined mesh data
    setupBuffers();
    
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    
    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        
        // Render the scene with broken offset calculations
        renderScene();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);
    
    glfwTerminate();
    return 0;
}