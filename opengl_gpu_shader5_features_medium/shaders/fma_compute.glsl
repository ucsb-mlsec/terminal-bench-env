#version 400
#extension GL_ARB_gpu_shader5 : enable

precision highp float;

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D texSampler;

void main() {
    // Basic texture sampling
    vec4 baseColor = texture(texSampler, vTexCoord);
    
    // ISSUE 1: Inconsistent precision across all three operands
    highp float a = 1.0;
    mediump float b = 2.0;
    lowp float c = 3.0;
    float result = fma(a, b, c);
    
    // Some intermediate calculations
    float factor = result * 0.5;
    
    // ISSUE 2: Mismatched precision between two operands
    highp float x = vTexCoord.x * 2.0;
    mediump float y = vTexCoord.y * 3.0;
    highp float z = 1.5;
    float blendFactor = fma(x, y, z);
    
    // Final color computation
    vec3 modulated = baseColor.rgb * factor;
    fragColor = vec4(modulated * blendFactor, baseColor.a);
}