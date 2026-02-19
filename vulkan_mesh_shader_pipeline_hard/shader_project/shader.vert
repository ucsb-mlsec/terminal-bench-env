#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;
layout(location = 2) out vec3 fragNormal;

void main() {
    gl_Position = vec4(inPosition, 1.0);
    fragColor = inColor;
    fragNormal = vec3(0.0, 0.0, 1.0);
}