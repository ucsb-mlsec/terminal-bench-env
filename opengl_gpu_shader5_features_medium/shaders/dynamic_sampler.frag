#version 400
#extension GL_ARB_gpu_shader5 : enable

// Dynamic sampler indexing shader - correct implementation
// This shader demonstrates proper precision handling with dynamic indexing

uniform sampler2D textures[4];

in vec2 vTexCoord;
in flat int index;

out vec4 fragColor;

void main()
{
    // Proper precision qualifiers for all variables
    highp vec4 sampledColor;
    highp vec2 adjustedCoord;
    
    // Adjust coordinates with proper precision
    adjustedCoord = vTexCoord * 1.0;
    
    // Dynamic indexing with proper precision
    sampledColor = texture(textures[index], adjustedCoord);
    
    // Output with proper precision
    fragColor = sampledColor;
}