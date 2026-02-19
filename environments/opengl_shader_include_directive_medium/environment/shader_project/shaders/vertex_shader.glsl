#version 330 core

// Vertex shader inputs
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

// Include directives - various issues present
#include "/common/transforms.glsl"
#include /common/utils.glsl
#include "/common/lighting_helpers.glsl"

// Outputs to fragment shader
out vec3 fragNormal;
out vec3 fragPos;

// Uniforms
uniform mat4 mvpMatrix;
uniform mat4 modelMatrix;

void main()
{
    // Apply transformation from included functions
    vec4 worldPos = applyTransform(modelMatrix, vec4(position, 1.0));
    fragPos = worldPos.xyz;
    
    // Compute normal using utility function
    fragNormal = computeNormal(modelMatrix, normal);
    
    // Final position calculation
    gl_Position = mvpMatrix * vec4(position, 1.0);
}