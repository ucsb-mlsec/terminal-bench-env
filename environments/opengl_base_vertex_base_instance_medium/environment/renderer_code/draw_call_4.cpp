#include <GL/gl.h>
#include <cstddef>

// Simple non-batched rendering implementation
// This approach uses the basic glDrawElements call without any base vertex or base instance support
// Each mesh must be rendered with a separate draw call, which is less efficient for batched rendering

class SimpleRenderer {
public:
    // Render a single mesh using basic indexed drawing
    // This does NOT support batched rendering with base vertex/instance offsets
    void renderMesh(GLuint vao, GLuint indexCount) {
        // Bind the vertex array object for this mesh
        glBindVertexArray(vao);
        
        // Basic draw call - no base vertex or base instance functionality
        // All vertex indices start from 0 in the currently bound vertex buffer
        // No instance offset is available since this is non-instanced rendering
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, (void*)0);
        
        // This approach requires:
        // - Separate VAO bindings for each mesh
        // - Separate draw calls for each object
        // - Cannot efficiently batch multiple meshes in shared buffers
    }
    
    // Render multiple meshes (inefficient approach)
    void renderMultipleMeshes(GLuint* vaos, GLuint* indexCounts, int meshCount) {
        // Must issue separate draw call for each mesh
        for (int i = 0; i < meshCount; i++) {
            glBindVertexArray(vaos[i]);
            
            // Basic glDrawElements - no batching capability
            glDrawElements(GL_TRIANGLES, indexCounts[i], GL_UNSIGNED_INT, (void*)0);
        }
        // This results in meshCount draw calls instead of a single batched call
    }
    
    // Render with index buffer offset (still not base vertex)
    void renderMeshWithOffset(GLuint vao, GLuint indexCount, GLuint indexOffset) {
        glBindVertexArray(vao);
        
        // Index buffer offset using pointer arithmetic
        // This only offsets where we read indices from, NOT the vertex offset
        // Still no base vertex or base instance support
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 
                      (void*)(indexOffset * sizeof(GLuint)));
    }
};

// Example usage showing the limitations
void renderScene() {
    SimpleRenderer renderer;
    
    // Each mesh needs its own VAO and separate draw call
    GLuint meshVAO1 = 1;
    GLuint meshVAO2 = 2;
    GLuint meshVAO3 = 3;
    
    // Render each mesh individually - inefficient for large scenes
    renderer.renderMesh(meshVAO1, 36);  // Cube with 36 indices
    renderer.renderMesh(meshVAO2, 24);  // Another object
    renderer.renderMesh(meshVAO3, 48);  // Third object
    
    // Total: 3 draw calls, 3 VAO bindings
    // Modern batched rendering would do this in 1 draw call
}