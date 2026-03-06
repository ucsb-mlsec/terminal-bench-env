#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D texSampler;

void main() {
    vec2 texCoord = vec2(0.5, 0.5);
    vec3 texColor = texture(texSampler, texCoord).rgb;
    
    vec3 result = fragColor * intensity;
    
    vec3 finalColor = result + texColor;
    
    outColor = finalColor;
}