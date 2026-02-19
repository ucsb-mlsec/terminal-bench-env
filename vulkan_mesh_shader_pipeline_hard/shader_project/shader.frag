#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(dot(normalize(fragNormal), lightDir), 0.0);
    
    vec3 color = fragColor.rgb * diffuse;
    color += fragTexCoord.x * 0.1;
    
    outColor = vec4(color, fragColor.a);
}