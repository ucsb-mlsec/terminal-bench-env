#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;

uniform mat4 mvpMatrix;

void main() {
    gl_Position = mvpMatrix * vec4(inPosition, 1.0);
    
    // Deliberate error: undeclared variable 'lightDir'
    fragColor = normalize(lightDir);
    
    // Additional shading calculation
    vec3 normalizedNormal = normalize(inNormal);
    float intensity = max(dot(normalizedNormal, vec3(0.0, 1.0, 0.0)), 0.0);
    
    // Deliberate error: missing semicolon
    fragColor = fragColor * intensity
    
    fragColor = clamp(fragColor, 0.0, 1.0);
}