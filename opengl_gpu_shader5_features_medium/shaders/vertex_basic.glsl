#version 400
#extension GL_ARB_gpu_shader5 : enable

// Basic vertex shader - reference implementation
in vec3 position;
in vec2 texCoord;

out vec2 vTexCoord;

uniform highp mat4 modelViewProjection;

void main()
{
    // Standard vertex transformation
    gl_Position = modelViewProjection * vec4(position, 1.0);
    
    // Pass through texture coordinates
    vTexCoord = texCoord;
}