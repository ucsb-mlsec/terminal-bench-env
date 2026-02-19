#include <vector>
#include <cmath>
#include <glm/glm.hpp>

// These functions generate the actual geometry data that gets combined into the shared vertex and index buffers

// Vertex structure matching the renderer
struct Vertex {
    float position[3];
    float normal[3];
    float color[4];
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
};

// Generate cube geometry
// Produces: 24 vertices (4 per face * 6 faces), 36 indices (2 triangles per face * 3 indices * 6 faces)
MeshData generateCubeData() {
    MeshData mesh;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);
    
    float halfSize = 0.5f;
    
    // Front face (red) - facing +Z
    mesh.vertices.push_back({{-halfSize, -halfSize, halfSize}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}});
    mesh.vertices.push_back({{halfSize, -halfSize, halfSize}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}});
    mesh.vertices.push_back({{halfSize, halfSize, halfSize}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}});
    mesh.vertices.push_back({{-halfSize, halfSize, halfSize}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}});
    
    // Back face (green) - facing -Z
    mesh.vertices.push_back({{halfSize, -halfSize, -halfSize}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
    mesh.vertices.push_back({{-halfSize, -halfSize, -halfSize}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
    mesh.vertices.push_back({{-halfSize, halfSize, -halfSize}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
    mesh.vertices.push_back({{halfSize, halfSize, -halfSize}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}});
    
    // Left face (blue) - facing -X
    mesh.vertices.push_back({{-halfSize, -halfSize, -halfSize}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
    mesh.vertices.push_back({{-halfSize, -halfSize, halfSize}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
    mesh.vertices.push_back({{-halfSize, halfSize, halfSize}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
    mesh.vertices.push_back({{-halfSize, halfSize, -halfSize}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
    
    // Right face (yellow) - facing +X
    mesh.vertices.push_back({{halfSize, -halfSize, halfSize}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}});
    mesh.vertices.push_back({{halfSize, -halfSize, -halfSize}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}});
    mesh.vertices.push_back({{halfSize, halfSize, -halfSize}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}});
    mesh.vertices.push_back({{halfSize, halfSize, halfSize}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f, 1.0f}});
    
    // Top face (cyan) - facing +Y
    mesh.vertices.push_back({{-halfSize, halfSize, halfSize}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 1.0f}});
    mesh.vertices.push_back({{halfSize, halfSize, halfSize}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 1.0f}});
    mesh.vertices.push_back({{halfSize, halfSize, -halfSize}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 1.0f}});
    mesh.vertices.push_back({{-halfSize, halfSize, -halfSize}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f, 1.0f}});
    
    // Bottom face (magenta) - facing -Y
    mesh.vertices.push_back({{-halfSize, -halfSize, -halfSize}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 1.0f}});
    mesh.vertices.push_back({{halfSize, -halfSize, -halfSize}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 1.0f}});
    mesh.vertices.push_back({{halfSize, -halfSize, halfSize}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 1.0f}});
    mesh.vertices.push_back({{-halfSize, -halfSize, halfSize}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 1.0f}});
    
    // Indices for all 6 faces (36 total)
    for (unsigned int i = 0; i < 6; i++) {
        unsigned int base = i * 4;
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 3);
    }
    
    return mesh;
}

// Generate pyramid geometry
// Produces: 18 vertices, 18 indices
MeshData generatePyramidData() {
    MeshData mesh;
    mesh.vertices.reserve(18);
    mesh.indices.reserve(18);
    
    float baseSize = 0.5f;
    float height = 1.0f;
    
    // Apex position
    float apexX = 0.0f, apexY = height, apexZ = 0.0f;
    
    // Base corners
    float baseY = 0.0f;
    glm::vec3 base0(-baseSize, baseY, -baseSize);
    glm::vec3 base1(baseSize, baseY, -baseSize);
    glm::vec3 base2(baseSize, baseY, baseSize);
    glm::vec3 base3(-baseSize, baseY, baseSize);
    glm::vec3 apex(apexX, apexY, apexZ);
    
    // Orange color variations
    float orange1[4] = {0.9f, 0.6f, 0.2f, 1.0f};
    float orange2[4] = {0.95f, 0.65f, 0.25f, 1.0f};
    float orange3[4] = {0.85f, 0.55f, 0.15f, 1.0f};
    
    // Front face (indices 0-2)
    glm::vec3 frontNormal = glm::normalize(glm::cross(base2 - base1, apex - base1));
    mesh.vertices.push_back({{base1.x, base1.y, base1.z}, {frontNormal.x, frontNormal.y, frontNormal.z}, {orange1[0], orange1[1], orange1[2], orange1[3]}});
    mesh.vertices.push_back({{base2.x, base2.y, base2.z}, {frontNormal.x, frontNormal.y, frontNormal.z}, {orange1[0], orange1[1], orange1[2], orange1[3]}});
    mesh.vertices.push_back({{apex.x, apex.y, apex.z}, {frontNormal.x, frontNormal.y, frontNormal.z}, {orange1[0], orange1[1], orange1[2], orange1[3]}});
    
    // Right face (indices 3-5)
    glm::vec3 rightNormal = glm::normalize(glm::cross(base3 - base2, apex - base2));
    mesh.vertices.push_back({{base2.x, base2.y, base2.z}, {rightNormal.x, rightNormal.y, rightNormal.z}, {orange2[0], orange2[1], orange2[2], orange2[3]}});
    mesh.vertices.push_back({{base3.x, base3.y, base3.z}, {rightNormal.x, rightNormal.y, rightNormal.z}, {orange2[0], orange2[1], orange2[2], orange2[3]}});
    mesh.vertices.push_back({{apex.x, apex.y, apex.z}, {rightNormal.x, rightNormal.y, rightNormal.z}, {orange2[0], orange2[1], orange2[2], orange2[3]}});
    
    // Back face (indices 6-8)
    glm::vec3 backNormal = glm::normalize(glm::cross(base0 - base3, apex - base3));
    mesh.vertices.push_back({{base3.x, base3.y, base3.z}, {backNormal.x, backNormal.y, backNormal.z}, {orange3[0], orange3[1], orange3[2], orange3[3]}});
    mesh.vertices.push_back({{base0.x, base0.y, base0.z}, {backNormal.x, backNormal.y, backNormal.z}, {orange3[0], orange3[1], orange3[2], orange3[3]}});
    mesh.vertices.push_back({{apex.x, apex.y, apex.z}, {backNormal.x, backNormal.y, backNormal.z}, {orange3[0], orange3[1], orange3[2], orange3[3]}});
    
    // Left face (indices 9-11)
    glm::vec3 leftNormal = glm::normalize(glm::cross(base1 - base0, apex - base0));
    mesh.vertices.push_back({{base0.x, base0.y, base0.z}, {leftNormal.x, leftNormal.y, leftNormal.z}, {orange1[0], orange1[1], orange1[2], orange1[3]}});
    mesh.vertices.push_back({{base1.x, base1.y, base1.z}, {leftNormal.x, leftNormal.y, leftNormal.z}, {orange1[0], orange1[1], orange1[2], orange1[3]}});
    mesh.vertices.push_back({{apex.x, apex.y, apex.z}, {leftNormal.x, leftNormal.y, leftNormal.z}, {orange1[0], orange1[1], orange1[2], orange1[3]}});
    
    // Base face (indices 12-17) - two triangles
    glm::vec3 baseNormal(0.0f, -1.0f, 0.0f);
    mesh.vertices.push_back({{base0.x, base0.y, base0.z}, {baseNormal.x, baseNormal.y, baseNormal.z}, {orange2[0], orange2[1], orange2[2], orange2[3]}});
    mesh.vertices.push_back({{base1.x, base1.y, base1.z}, {baseNormal.x, baseNormal.y, baseNormal.z}, {orange2[0], orange2[1], orange2[2], orange2[3]}});
    mesh.vertices.push_back({{base2.x, base2.y, base2.z}, {baseNormal.x, baseNormal.y, baseNormal.z}, {orange2[0], orange2[1], orange2[2], orange2[3]}});
    mesh.vertices.push_back({{base0.x, base0.y, base0.z}, {baseNormal.x, baseNormal.y, baseNormal.z}, {orange2[0], orange2[1], orange2[2], orange2[3]}});
    mesh.vertices.push_back({{base2.x, base2.y, base2.z}, {baseNormal.x, baseNormal.y, baseNormal.z}, {orange2[0], orange2[1], orange2[2], orange2[3]}});
    mesh.vertices.push_back({{base3.x, base3.y, base3.z}, {baseNormal.x, baseNormal.y, baseNormal.z}, {orange2[0], orange2[1], orange2[2], orange2[3]}});
    
    // Indices (18 total)
    for (unsigned int i = 0; i < 18; i++) {
        mesh.indices.push_back(i);
    }
    
    return mesh;
}

// Generate low-poly sphere geometry
// Produces: 32 vertices, 60 indices
MeshData generateSphereData() {
    MeshData mesh;
    mesh.vertices.reserve(32);
    mesh.indices.reserve(60);
    
    float radius = 0.5f;
    int rings = 4;  // Latitude divisions
    int segments = 8;  // Longitude divisions
    
    // Purple/pink gradient colors
    float color1[4] = {0.7f, 0.3f, 0.8f, 1.0f};
    float color2[4] = {0.9f, 0.5f, 0.9f, 1.0f};
    
    // Generate vertices
    // Top pole
    mesh.vertices.push_back({{0.0f, radius, 0.0f}, {0.0f, 1.0f, 0.0f}, {color2[0], color2[1], color2[2], color2[3]}});
    
    // Middle rings (3 rings * 8 segments = 24 vertices)
    for (int r = 1; r < rings; r++) {
        float phi = M_PI * float(r) / float(rings);
        float y = radius * cosf(phi);
        float ringRadius = radius * sinf(phi);
        
        for (int s = 0; s < segments; s++) {
            float theta = 2.0f * M_PI * float(s) / float(segments);
            float x = ringRadius * cosf(theta);
            float z = ringRadius * sinf(theta);
            
            // Normal is the normalized position for a sphere
            glm::vec3 normal = glm::normalize(glm::vec3(x, y, z));
            
            // Interpolate color based on height
            float t = (y + radius) / (2.0f * radius);
            float color[4] = {
                color1[0] + t * (color2[0] - color1[0]),
                color1[1] + t * (color2[1] - color1[1]),
                color1[2] + t * (color2[2] - color1[2]),
                1.0f
            };
            
            mesh.vertices.push_back({{x, y, z}, {normal.x, normal.y, normal.z}, {color[0], color[1], color[2], color[3]}});
        }
    }
    
    // Bottom pole (vertex 25)
    mesh.vertices.push_back({{0.0f, -radius, 0.0f}, {0.0f, -1.0f, 0.0f}, {color1[0], color1[1], color1[2], color1[3]}});
    
    // Add 6 more vertices to reach exactly 32 (duplicate some for cleaner indexing)
    for (int i = 0; i < 6; i++) {
        float theta = 2.0f * M_PI * float(i) / 6.0f;
        float x = radius * 0.7f * cosf(theta);
        float z = radius * 0.7f * sinf(theta);
        glm::vec3 normal = glm::normalize(glm::vec3(x, 0.0f, z));
        mesh.vertices.push_back({{x, 0.0f, z}, {normal.x, normal.y, normal.z}, {color1[0], color1[1], color1[2], color1[3]}});
    }
    
    // Generate indices for top cap (8 triangles)
    for (int s = 0; s < segments; s++) {
        int next = (s + 1) % segments;
        mesh.indices.push_back(0);
        mesh.indices.push_back(1 + s);
        mesh.indices.push_back(1 + next);
    }
    
    // Generate indices for middle rings (2 rings * 8 segments * 2 triangles = 32 triangles = 96 indices, but we limit to fit 60)
    // Actually generate fewer quads to fit our 60-index limit: 10 quads = 60 indices
    for (int q = 0; q < 10; q++) {
        int r = q / 5;
        int s = (q * 4) % segments;
        
        if (r < rings - 2) {
            int current = 1 + r * segments + s;
            int next = 1 + r * segments + ((s + 1) % segments);
            int currentBelow = 1 + (r + 1) * segments + s;
            int nextBelow = 1 + (r + 1) * segments + ((s + 1) % segments);
            
            mesh.indices.push_back(current);
            mesh.indices.push_back(currentBelow);
            mesh.indices.push_back(next);
            mesh.indices.push_back(next);
            mesh.indices.push_back(currentBelow);
            mesh.indices.push_back(nextBelow);
        }
    }
    
    return mesh;
}

// Combine all mesh data into single buffers
// Returns combined vertex buffer, combined index buffer, and offset information
struct CombinedBuffers {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    
    // Offsets for each mesh type
    unsigned int cubeBaseVertex;
    unsigned int cubeBaseIndex;
    unsigned int pyramidBaseVertex;
    unsigned int pyramidBaseIndex;
    unsigned int sphereBaseVertex;
    unsigned int sphereBaseIndex;
};

CombinedBuffers combineBuffers() {
    CombinedBuffers combined;
    
    // Generate individual mesh data
    MeshData cube = generateCubeData();
    MeshData pyramid = generatePyramidData();
    MeshData sphere = generateSphereData();
    
    // Cube goes first
    // Cube starts at vertex 0, index 0
    combined.cubeBaseVertex = 0;
    combined.cubeBaseIndex = 0;
    
    combined.vertices.insert(combined.vertices.end(), cube.vertices.begin(), cube.vertices.end());
    combined.indices.insert(combined.indices.end(), cube.indices.begin(), cube.indices.end());
    
    // Pyramid goes second
    // Pyramid starts at vertex 24 (after cube's 24 vertices), index 36 (after cube's 36 indices)
    combined.pyramidBaseVertex = combined.vertices.size();
    combined.pyramidBaseIndex = combined.indices.size();
    
    // Adjust pyramid indices to reference correct vertices
    for (unsigned int idx : pyramid.indices) {
        combined.indices.push_back(idx + combined.pyramidBaseVertex);
    }
    combined.vertices.insert(combined.vertices.end(), pyramid.vertices.begin(), pyramid.vertices.end());
    
    // Sphere goes third
    // Sphere starts at vertex 42 (24 + 18), index 54 (36 + 18)
    combined.sphereBaseVertex = combined.vertices.size();
    combined.sphereBaseIndex = combined.indices.size();
    
    // Adjust sphere indices to reference correct vertices
    for (unsigned int idx : sphere.indices) {
        combined.indices.push_back(idx + combined.sphereBaseVertex);
    }
    combined.vertices.insert(combined.vertices.end(), sphere.vertices.begin(), sphere.vertices.end());
    
    return combined;
}