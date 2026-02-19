#version 330 core

// Input vertex attributes
in vec3 vertex_position;
in vec3 vertex_normal;
in vec2 texcoord;

// Output variables for transform feedback
out vec3 world_position;
out vec3 world_normal;
out vec2 uv_coords;
out float depth_value;

// Transformation matrices
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

void main()
{
    // Transform vertex position to world space
    vec4 worldPos = modelMatrix * vec4(vertex_position, 1.0);
    world_position = worldPos.xyz;
    
    // Transform normal to world space (using normal matrix)
    mat3 normalMatrix = mat3(transpose(inverse(modelMatrix)));
    world_normal = normalize(normalMatrix * vertex_normal);
    
    // Pass through texture coordinates
    uv_coords = texcoord;
    
    // Calculate view space position for depth
    vec4 viewPos = viewMatrix * worldPos;
    depth_value = -viewPos.z;
    
    // Calculate final clip space position
    gl_Position = projectionMatrix * viewPos;
}