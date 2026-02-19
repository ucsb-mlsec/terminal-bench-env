#include <GL/gl.h>
#include <cstddef>

// Batched mesh renderer using base vertex offset
// This allows multiple meshes to be stored in a single vertex buffer
// and rendered with appropriate offsets

class BatchedRenderer {
public:
    // Render a mesh from a combined vertex/index buffer
    // This function demonstrates base vertex offset usage
    void renderMeshes() {
        // Assume we have already bound the VAO, vertex buffer, and index buffer
        
        // Mesh 1: Cube at the start of the buffer
        // Indices: 0-35 (36 indices)
        // Vertices: 0-23 (24 vertices)
        int mesh1_indexCount = 36;
        int mesh1_indexOffset = 0;
        int mesh1_baseVertex = 0;
        
        glDrawElementsBaseVertex(
            GL_TRIANGLES,                                    // primitive type
            mesh1_indexCount,                                // number of indices
            GL_UNSIGNED_INT,                                 // index type
            (void*)(mesh1_indexOffset * sizeof(unsigned int)), // offset into index buffer
            mesh1_baseVertex                                 // base vertex offset
        );
        
        // Mesh 2: Sphere stored after the cube
        // Indices: 36-935 (900 indices)
        // Vertices: 24-323 (300 vertices, starting at vertex 24)
        int mesh2_indexCount = 900;
        int mesh2_indexOffset = 36;
        int mesh2_baseVertex = 24;  // Non-zero base vertex offset
        
        // This is the key feature: baseVertex=24 means all indices in the index buffer
        // will have 24 added to them when fetching vertices
        // So index buffer value "0" actually fetches vertex 24, "1" fetches vertex 25, etc.
        glDrawElementsBaseVertex(
            GL_TRIANGLES,
            mesh2_indexCount,
            GL_UNSIGNED_INT,
            (void*)(mesh2_indexOffset * sizeof(unsigned int)),
            mesh2_baseVertex  // Base vertex allows reusing index ranges
        );
        
        // Mesh 3: Cylinder stored after the sphere
        // Indices: 936-1235 (300 indices)
        // Vertices: 324-423 (100 vertices)
        int mesh3_indexCount = 300;
        int mesh3_indexOffset = 936;
        int mesh3_baseVertex = 324;  // Another non-zero base vertex offset
        
        glDrawElementsBaseVertex(
            GL_TRIANGLES,
            mesh3_indexCount,
            GL_UNSIGNED_INT,
            (void*)(mesh3_indexOffset * sizeof(unsigned int)),
            mesh3_baseVertex
        );
        
        // Benefits of using base vertex:
        // 1. All meshes share one vertex buffer and one index buffer
        // 2. Reduces buffer binding overhead
        // 3. Index data for each mesh can start from 0, making it easier to manage
        // 4. Better cache coherency and reduced state changes
    }
    
    void renderSingleMesh(int indexCount, int indexOffset, int baseVertexOffset) {
        // Helper function for rendering a single mesh with base vertex offset
        glDrawElementsBaseVertex(
            GL_TRIANGLES,
            indexCount,
            GL_UNSIGNED_INT,
            (void*)(indexOffset * sizeof(unsigned int)),
            baseVertexOffset  // This is the base vertex parameter
        );
    }
};