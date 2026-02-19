#version 450

layout(location = 0) in vec3 particleCenter;
layout(location = 1) in vec2 cornerOffset;

layout(location = 0) out vec2 fragUV;
out vec3 worldPosition;

uniform mat4 viewMatrix;
uniform mat4 projMatrix;

void main() {
    // Extract right and up vectors from view matrix for billboard orientation
    vec3 cameraRight = vec3(viewMatrix[0].x, viewMatrix[1].x, viewMatrix[2].x);
    vec3 cameraUp = vec3(viewMatrix[0].y, viewMatrix[1].y, viewMatrix[2].y);
    
    // Calculate billboard corner position in world space
    vec3 vertexPosition = particleCenter + 
                         billboardRight * cornerOffset.x + 
                         billboardUp * cornerOffset.y;
    
    // Calculate final position
    vec4 viewPos = viewMatrix * vec4(vertexPosition, 1.0);
    gl_Position = projMatrix * viewPos;
    
    // Pass UV coordinates to fragment shader
    fragUV = cornerOffset * 0.5 + 0.5;
    
    // Store world position for lighting calculations
    worldPosition = vertexPosition.xyw
}