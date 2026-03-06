// renderer_snippet.cpp
// Graphics Rendering Library - Indexed Drawing Module
// Renders multiple disconnected geometric shapes using indexed drawing

#include <GL/gl.h>
#include <vector>

// Vertex structure for our geometry
struct Vertex {
    float x, y, z;
    float r, g, b;
};

// Global buffer handles
GLuint vertexBuffer = 0;
GLuint indexBuffer = 0;

// Function to render multiple disconnected shapes using indexed drawing
// This allows efficient rendering of separate shapes in a single draw call
void renderMultipleShapes() {
    // Vertex data for multiple triangles and quads
    std::vector<Vertex> vertices = {
        // First triangle (indices 0-2)
        {-0.8f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f},
        {-0.6f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f},
        {-0.7f, -0.3f, 0.0f, 1.0f, 0.0f, 0.0f},
        
        // Second triangle (indices 3-5)
        {-0.3f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f},
        {-0.1f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f},
        {-0.2f, -0.3f, 0.0f, 0.0f, 1.0f, 0.0f},
        
        // Third triangle (indices 6-8)
        {0.3f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f},
        {0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f},
        {0.4f, -0.3f, 0.0f, 0.0f, 0.0f, 1.0f},
    };
    
    // Index buffer with primitive restart markers
    // Using 0xFFFFFFFF (4294967295 in decimal) as the separator marker
    // to indicate where one primitive ends and the next begins
    std::vector<GLuint> indices = {
        // First triangle
        0, 1, 2,
        // Separator marker - indicates end of first primitive
        0xFFFFFFFF,
        // Second triangle
        3, 4, 5,
        // Separator marker - indicates end of second primitive
        0xFFFFFFFF,
        // Third triangle
        6, 7, 8
    };
    
    // Generate and bind vertex buffer
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), 
                 vertices.data(), GL_STATIC_DRAW);
    
    // Generate and bind index buffer
    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), 
                 indices.data(), GL_STATIC_DRAW);
    
    // Set up vertex attributes
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
                         (void*)0);
    
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 
                         (void*)(3 * sizeof(float)));
    
    // Configure the primitive restart index value
    // This tells OpenGL which value in the index buffer should act as a separator
    // We're using 0xFFFFFFFF (decimal: 4294967295) as the marker value
    glPrimitiveRestartIndex(0xFFFFFFFF);
    
    // TODO: Missing critical OpenGL state configuration here!
    // The primitive restart markers in the index buffer won't work without
    // enabling the corresponding OpenGL capability
    
    // Perform the indexed draw call
    // This should render three separate triangles, but without the proper
    // OpenGL state enabled, the separator markers (0xFFFFFFFF) will be
    // treated as regular indices, causing rendering artifacts
    glDrawElements(GL_TRIANGLE_STRIP, indices.size(), GL_UNSIGNED_INT, 0);
    
    // Clean up
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
}

// Cleanup function
void cleanupBuffers() {
    if (vertexBuffer != 0) {
        glDeleteBuffers(1, &vertexBuffer);
        vertexBuffer = 0;
    }
    if (indexBuffer != 0) {
        glDeleteBuffers(1, &indexBuffer);
        indexBuffer = 0;
    }
}