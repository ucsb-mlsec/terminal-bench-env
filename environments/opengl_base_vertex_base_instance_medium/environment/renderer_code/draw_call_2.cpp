#include <GL/gl.h>
#include <cstddef>

// Mesh Renderer - Instanced Drawing Implementation
// This renderer attempts to batch multiple objects using instanced rendering

class MeshBatch {
public:
    GLuint vao;
    GLuint vertexBuffer;
    GLuint indexBuffer;
    int indexCount;
    int instanceCount;
    
    // Render multiple instances of the mesh
    // INCORRECT: Attempting to use base instance offset with wrong function
    void render() {
        glBindVertexArray(vao);
        
        // ISSUE: This code tries to pass a base instance parameter
        // but glDrawElementsInstanced doesn't support base instance offset!
        // The function signature is:
        // glDrawElementsInstanced(mode, count, type, indices, instancecount)
        // 
        // The developer is incorrectly trying to offset instance IDs
        // by casting an integer to void* for the indices parameter
        
        int baseInstance = 100;  // Trying to offset instance IDs
        
        // WRONG: Attempting to use baseInstance as the indices offset
        // This will be interpreted as a byte offset into the index buffer,
        // not as an instance ID offset!
        glDrawElementsInstanced(
            GL_TRIANGLES, 
            indexCount, 
            GL_UNSIGNED_INT, 
            (void*)(baseInstance * sizeof(GLuint)),  // INCORRECT: misusing indices parameter
            instanceCount
        );
        
        // The correct approach would be to use glDrawElementsInstancedBaseInstance
        // or handle the offset in the shader using gl_InstanceID
    }
    
    // Another incorrect attempt in a different function
    void renderWithOffset(int instanceOffset) {
        glBindVertexArray(vao);
        
        // WRONG: Parameters in incorrect order
        // Developer swapped instanceCount and type parameters
        glDrawElementsInstanced(
            GL_TRIANGLES,
            indexCount,
            instanceCount,  // WRONG: this should be GL_UNSIGNED_INT (type)
            (void*)0,
            GL_UNSIGNED_INT  // WRONG: this should be instanceCount (integer)
        );
        
        // This will likely cause a compilation error or runtime crash
    }
};