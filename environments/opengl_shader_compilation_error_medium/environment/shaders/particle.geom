#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 4.5) out;

in vec3 Position[];

out vec2 TexCoord;

uniform mat4 projection;

void main()
{
    vec4 basePos = vec4(Position[0], 1.0);
    
    float size = 0.5;
    
    // Bottom-left vertex
    gl_Position = projection * (basePos + vec4(-size, -size, 0.0, 0.0));
    TexCoord = vec2(0.0, 0.0);
    EmitVertex();
    
    // Bottom-right vertex
    gl_Position = projection * (basePos + vec4(size, -size, 0.0, 0.0));
    TexCoord = vec2(1.0, 0.0);
    EmitVertex();
    
    // Top-left vertex
    gl_Position = projection * (basePos + vec4(-size, size, 0.0, 0.0));
    TexCoord = vec2(0.0, 1.0);
    EmitVertex();
    
    EndPrimitive();
}