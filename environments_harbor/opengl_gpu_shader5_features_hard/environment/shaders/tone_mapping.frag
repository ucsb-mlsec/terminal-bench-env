#version 150
#extension GL_ARB_gpu_shader5 : enable

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D hdrTexture;
uniform float exposure;
uniform float gamma;
uniform float whitePoint;

void main()
{
    // Sample HDR color from texture
    vec3 hdrColor = texture(hdrTexture, texCoord).rgb;
    
    // Apply exposure adjustment using separate multiply
    vec3 exposed = hdrColor * exposure;
    
    // Reinhard tone mapping operator
    vec3 mapped = exposed / (exposed + vec3(1.0));
    
    // White point adjustment
    float whiteScale = 1.0 / whitePoint;
    vec3 whiteMapped = mapped * whiteScale;
    
    // Gamma correction with offset - using separate operations
    vec3 offset = vec3(0.05);
    vec3 gammaAdjusted = whiteMapped * gamma;
    vec3 gammaCorrect = gammaAdjusted + offset;
    
    // Final color adjustment
    vec3 finalColor = gammaCorrect * 0.95 + vec3(0.01);
    
    // Output with alpha
    fragColor = vec4(finalColor, 1.0);
}