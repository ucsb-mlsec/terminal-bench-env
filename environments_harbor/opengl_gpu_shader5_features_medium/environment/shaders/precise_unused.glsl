#version 400
#extension GL_ARB_gpu_shader5 : enable

// Fragment shader with unused precise qualifiers

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D mainTexture;
uniform float brightness;

// ISSUE 1: precise variable that is never used
precise float unusedVar = 1.0;

// ISSUE 2: precise variable used only in simple operations
precise vec3 simpleColor;

void main()
{
    // Simple texture sampling
    vec4 texColor = texture(mainTexture, vTexCoord);
    
    // ISSUE 2: simpleColor assigned constant and used in basic addition
    simpleColor = vec3(0.5, 0.5, 0.5);
    vec3 adjustedColor = texColor.rgb + simpleColor;
    
    // Basic brightness adjustment
    adjustedColor *= brightness;
    
    fragColor = vec4(adjustedColor, texColor.a);
}