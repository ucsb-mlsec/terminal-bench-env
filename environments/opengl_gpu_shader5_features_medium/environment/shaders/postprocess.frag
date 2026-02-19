#version 400
#extension GL_ARB_gpu_shader5 : enable

// Post-processing shader - simple blur effect
// Single-pass rendering, no precision issues

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D sceneTexture;
uniform highp float blurAmount;

void main()
{
    highp vec2 texelSize = vec2(1.0) / vec2(textureSize(sceneTexture, 0));
    
    highp vec4 color = vec4(0.0);
    highp float weight = 0.0;
    
    // Simple 3x3 blur kernel
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            highp vec2 offset = vec2(float(x), float(y)) * texelSize * blurAmount;
            highp float kernelWeight = 1.0 / 9.0;
            color += texture(sceneTexture, vTexCoord + offset) * kernelWeight;
            weight += kernelWeight;
        }
    }
    
    fragColor = color / weight;
}