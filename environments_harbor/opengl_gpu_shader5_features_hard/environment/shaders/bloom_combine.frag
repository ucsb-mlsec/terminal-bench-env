#version 150
#extension GL_ARB_gpu_shader5 : enable

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D sceneTexture;
uniform sampler2D bloomTexture;
uniform float bloomStrength;
uniform float exposure;

void main()
{
    vec3 scene = texture(sceneTexture, texCoord).rgb;
    vec3 bloom = texture(bloomTexture, texCoord).rgb;
    
    vec3 combined = fma(bloom, vec3(bloomStrength), scene);
    vec3 result = fma(combined, vec3(exposure), vec3(0.0));
    
    fragColor = vec4(result, 1.0);
}