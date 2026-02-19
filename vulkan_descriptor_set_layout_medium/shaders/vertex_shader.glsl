#version 450

// Descriptor bindings
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
    vec4 lightPosition;
    float time;
} ubo;

layout(set = 0, binding = 1) uniform sampler2D displacementMap;

layout(set = 0, binding = 2) uniform sampler2D heightMap;

// Input vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;

// Output variables to fragment shader
layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 fragViewPos;
layout(location = 4) out float fragDisplacement;

void main() {
    // Sample displacement from texture
    float displacement = texture(displacementMap, inTexCoord).r;
    float heightValue = texture(heightMap, inTexCoord).r;
    
    // Apply displacement along normal
    vec3 displacedPosition = inPosition + inNormal * displacement * 0.5;
    displacedPosition += inNormal * heightValue * 0.3;
    
    // Transform position
    vec4 worldPosition = ubo.model * vec4(displacedPosition, 1.0);
    vec4 viewPosition = ubo.view * worldPosition;
    gl_Position = ubo.projection * viewPosition;
    
    // Calculate world space normal
    mat3 normalMatrix = transpose(inverse(mat3(ubo.model)));
    vec3 worldNormal = normalize(normalMatrix * inNormal);
    
    // Pass data to fragment shader
    fragPosition = worldPosition.xyz;
    fragNormal = worldNormal;
    fragTexCoord = inTexCoord;
    fragViewPos = viewPosition.xyz;
    fragDisplacement = displacement;
    
    // Apply time-based animation
    gl_Position.y += sin(ubo.time + inPosition.x) * 0.1;
}