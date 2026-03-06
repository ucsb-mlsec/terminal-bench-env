#include <GL/gl.h>
#include <cstddef>

// BatchedInstancedRenderer: Demonstrates correct usage of base vertex and base instance
// for efficient batched instanced rendering of multiple meshes from shared buffers.
//
// This renderer combines multiple different meshes into a single vertex buffer and
// index buffer, and uses base vertex/base instance offsets to render different
// mesh instances from the combined buffers in a single draw call.

class BatchedInstancedRenderer {
public:
    // Renders a batch of instanced meshes using advanced OpenGL features
    // 
    // Parameters:
    //   indexCount: Number of indices to draw for this mesh
    //   indexOffset: Offset into the index buffer (in index units, not bytes)
    //   instanceCount: Number of instances to render
    //   baseVertex: Offset added to each index before accessing vertex buffer
    //   baseInstance: Offset added to gl_InstanceID for shader usage
    void renderBatchedInstances(int indexCount, int indexOffset, int instanceCount, 
                                 int baseVertex, int baseInstance) {
        // Use glDrawElementsInstancedBaseVertexBaseInstance for maximum flexibility
        // This allows us to:
        // 1. Draw multiple instances (instancing)
        // 2. Offset vertex indices (base vertex) - useful when multiple meshes share a VBO
        // 3. Offset instance IDs (base instance) - useful for accessing instance data arrays
        //
        // Example scenario: We have 3 different meshes combined in one buffer:
        // - Mesh A: vertices 0-99, indices 0-299
        // - Mesh B: vertices 100-227, indices 300-899  
        // - Mesh C: vertices 228-455, indices 900-1499
        //
        // To draw Mesh B with 64 instances starting at instance data index 256:
        // - indexCount: 600 (number of indices for Mesh B)
        // - indexOffset: 300 (start of Mesh B indices)
        // - instanceCount: 64 (render 64 instances)
        // - baseVertex: 100 (Mesh B starts at vertex 100)
        // - baseInstance: 256 (instance data starts at index 256)
        
        int meshIndexCount = 600;
        int meshIndexOffset = 300;
        int meshInstanceCount = 64;
        int meshBaseVertex = 128;      // Non-zero offset for vertex data
        int meshBaseInstance = 256;    // Non-zero offset for instance data
        
        // This is the correct function call with all parameters in proper order:
        // GLenum mode, GLsizei count, GLenum type, const void *indices, 
        // GLsizei instancecount, GLint basevertex, GLuint baseinstance
        glDrawElementsInstancedBaseVertexBaseInstance(
            GL_TRIANGLES,                                      // Primitive type
            meshIndexCount,                                    // Number of indices
            GL_UNSIGNED_INT,                                   // Index type
            (void*)(meshIndexOffset * sizeof(unsigned int)),  // Index buffer offset (in bytes)
            meshInstanceCount,                                 // Number of instances
            meshBaseVertex,                                    // Base vertex offset
            meshBaseInstance                                   // Base instance offset
        );
        
        // The shader will receive:
        // - Vertex indices offset by meshBaseVertex (128)
        // - gl_InstanceID values starting from meshBaseInstance (256)
        //
        // This enables efficient rendering of multiple different meshes with instancing,
        // all from shared vertex and index buffers, in a single draw call.
    }
    
    // Example usage for rendering multiple batched mesh groups
    void renderScene() {
        // Render first mesh batch: 32 instances of mesh at vertices 0-127
        glDrawElementsInstancedBaseVertexBaseInstance(
            GL_TRIANGLES, 
            384,                                    // Index count for this mesh
            GL_UNSIGNED_INT, 
            (void*)(0 * sizeof(unsigned int)),     // Start at beginning of index buffer
            32,                                     // 32 instances
            0,                                      // Vertices start at 0
            0                                       // Instance data starts at 0
        );
        
        // Render second mesh batch: 64 instances of mesh at vertices 128-255
        glDrawElementsInstancedBaseVertexBaseInstance(
            GL_TRIANGLES,
            600,                                    // Different index count
            GL_UNSIGNED_INT,
            (void*)(384 * sizeof(unsigned int)),   // Continue after first mesh indices
            64,                                     // 64 instances
            128,                                    // Vertices start at 128 (non-zero base)
            32                                      // Instance data starts at 32 (after first batch)
        );
        
        // Render third mesh batch: 128 instances of mesh at vertices 256-511
        glDrawElementsInstancedBaseVertexBaseInstance(
            GL_TRIANGLES,
            900,
            GL_UNSIGNED_INT,
            (void*)(984 * sizeof(unsigned int)),
            128,
            256,                                    // Base vertex of 256
            96                                      // Base instance of 96 (32 + 64 from previous batches)
        );
    }
};