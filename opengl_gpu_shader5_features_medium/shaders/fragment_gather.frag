#version 400
#extension GL_ARB_gpu_shader5 : enable

// Fragment shader for texture gather operations

uniform sampler2D colorTexture;
uniform sampler2D normalTexture;

in vec2 vTexCoord;

out vec4 fragColor;

void main()
{
    // ISSUE 1: Missing precision qualifier on gather result
    vec4 gatherResult = textureGather(colorTexture, vTexCoord, 0);
    
    // ISSUE 2: Missing precision qualifier on another gather variable
    vec4 normalGather = textureGather(normalTexture, vTexCoord, 2);
    
    // Process gathered samples
    float avgColor = (gatherResult.x + gatherResult.y + gatherResult.z + gatherResult.w) * 0.25;
    float avgNormal = (normalGather.x + normalGather.y + normalGather.z + normalGather.w) * 0.25;
    
    // Combine results
    fragColor = vec4(avgColor, avgNormal, 0.0, 1.0);
}