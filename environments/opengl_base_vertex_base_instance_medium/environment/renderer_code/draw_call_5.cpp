#include <GL/gl.h>
#include <cstddef>

// Renderer implementation using instanced rendering with base vertex offset
// This approach allows multiple instances of a mesh to be drawn with a single call
// while supporting vertex buffer offset for batched geometry

class BatchedInstancedRenderer {
public:
    // Draws multiple instances of a mesh using base vertex offset
    // This is useful when multiple meshes are packed into a single vertex buffer
    // and we want to draw multiple instances of one specific mesh
    void renderMeshInstanced(int indexCount, int indexOffset, int instanceCount, int baseVertex) {
        // Use glDrawElementsInstancedBaseVertex for instanced rendering with vertex offset
        // This function combines instancing with base vertex offset capability
        // 
        // Parameters:
        // - GL_TRIANGLES: primitive type
        // - indexCount: number of indices to draw
        // - GL_UNSIGNED_INT: type of indices
        // - (void*)(indexOffset * sizeof(unsigned int)): byte offset into index buffer
        // - instanceCount: number of instances to render
        // - baseVertex: offset added to each vertex index before reading from vertex buffer
        //
        // Note: This function does NOT support baseInstance parameter
        // All instances will have gl_InstanceID starting from 0
        // For baseInstance support, use glDrawElementsInstancedBaseVertexBaseInstance instead
        
        glDrawElementsInstancedBaseVertex(
            GL_TRIANGLES,                                    // mode
            indexCount,                                      // count
            GL_UNSIGNED_INT,                                 // type
            (void*)(indexOffset * sizeof(unsigned int)),    // indices offset
            instanceCount,                                   // instancecount
            baseVertex                                       // basevertex
        );
    }
    
    // Example usage: Draw a batched mesh with vertex offset
    void drawBatchedMesh() {
        // Assume we have multiple meshes packed into one vertex buffer:
        // Mesh 0: vertices 0-511
        // Mesh 1: vertices 512-1023  <- We want to draw this one instanced
        // Mesh 2: vertices 1024-1535
        
        int indexCount = 360;      // 120 triangles
        int indexOffset = 1000;    // Start at index 1000 in the index buffer
        int instanceCount = 50;    // Draw 50 instances of this mesh
        int baseVertex = 512;      // Offset to start of mesh 1 in vertex buffer
        
        // This will draw 50 instances of the mesh starting at vertex 512
        // Each instance can be positioned/transformed differently using gl_InstanceID
        // in the vertex shader to index into a transform buffer
        renderMeshInstanced(indexCount, indexOffset, instanceCount, baseVertex);
    }
    
    // Alternative rendering scenario: multiple meshes from a batched buffer
    void drawMultipleBatchedMeshes() {
        // Draw mesh at vertex offset 0 with 10 instances
        renderMeshInstanced(180, 0, 10, 0);
        
        // Draw mesh at vertex offset 512 with 25 instances
        renderMeshInstanced(360, 500, 25, 512);
        
        // Draw mesh at vertex offset 1024 with 5 instances
        renderMeshInstanced(540, 1200, 5, 1024);
        
        // This demonstrates batched geometry where each mesh lives at a different
        // vertex offset in the shared vertex buffer, and each is drawn with instancing
    }
};