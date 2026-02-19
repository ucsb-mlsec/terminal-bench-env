#version 400
#extension GL_ARB_gpu_shader5 : enable

// Fragment shader demonstrating fused multiply-add operations

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D mainTexture;

void main()
{
    // Correct fma usage - consistent highp precision
    highp float a = 1.0;
    highp float b = 2.0;
    highp float c = 3.0;
    highp float good = fma(a, b, c);
    
    // ISSUE: Inconsistent precision in fma operands
    highp float x = 1.0;
    mediump float y = 2.0;
    highp float z = 3.0;
    float bad = fma(x, y, z);
    
    // Sample texture and apply calculations
    vec4 texColor = texture(mainTexture, vTexCoord);
    
    // Use fma results in fragment calculation
    float intensity = good * 0.1 + bad * 0.05;
    
    fragColor = texColor * intensity;
}