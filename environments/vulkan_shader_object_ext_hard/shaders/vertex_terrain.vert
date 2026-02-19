#version 450

// Input vertex attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

// Output variables to fragment shader
layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec2 fragUV;
out vec3 fragNormal;  // Missing layout location

// Uniform buffer for transformation matrices
layout(set = 0, binding = 0) uniform MatrixUBO {
    mat4 model;
    mat4 view;
    mat4 projection;
} matrices;

// Uniform buffer for terrain parameters
layout(set = 0, binding = 1) uniform TerrainParams {
    float heightScale;
    float uvScale;
    vec2 terrainSize;
} terrain;

// Sampler for height map is NOT declared but used below

void main() {
    // Scale texture coordinates
    vec2 scaledUV = texCoord * terrain.uvScale;
    fragUV = scaledUV;
    
    // Sample height from heightMap (ERROR: heightMap not declared)
    float height = texture(heightMap, scaledUV).r * terrain.heightScale;
    
    // Calculate world position with height offset
    vec3 worldPos = position;
    worldPos.y += height;
    
    // Transform position
    vec4 viewPos = matrices.view * matrices.model * vec4(worldPos, 1.0);
    gl_Position = matrices.projection * viewPos;
    
    // ERROR: fragPos should be vec3 but assigned vec2
    fragPos = scaledUV;
    
    // ERROR: calculateNormal() function is not defined
    fragNormal = calculateNormal(scaledUV, height);
    
    // ERROR: Incorrect layout qualifier format on line with binding
    // (This error is implicit - the sampler declaration would have wrong format if it existed)
}