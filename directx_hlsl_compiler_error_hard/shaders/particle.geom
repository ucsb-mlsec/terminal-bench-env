#version 330 core

in vec4 vertColor[];
in float vertSize[];

out vec3 fragColor;
out vec2 texCoord;

void main() {
    vec3 pos = gl_in[0].gl_Position.xyz;
    fragColor = vertColor[0].rgb;
    
    float size = particleSize * 0.5;
    
    // Bottom-left vertex
    gl_Position = vec4(pos.x - size, pos.y - size, pos.z, 1.0);
    texCoord = vec2(0.0, 0.0);
    EmitVertex();
    
    // Bottom-right vertex
    gl_Position = vec4(pos.x + size, pos.y - size, pos.z, 1.0);
    texCoord = vec2(1.0, 0.0);
    EmitVertex();
    
    // Top-left vertex
    gl_Position = vec4(pos.x - size, pos.y + size, pos.z, 1.0);
    texCoord = vec2(0.0, 1.0);
    EmitVertex();
    
    // Top-right vertex
    texCoord = vec2(1.0, 1.0);
    EmitVertex();
    
    EndPrimitive();
}