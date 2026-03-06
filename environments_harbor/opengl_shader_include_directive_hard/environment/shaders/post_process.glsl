// Post-processing shader for final image effects
#version 450 core

#include "filters/tonemap.glsl"
#include "common/utils.glsl"

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uSceneTexture;
uniform float uExposure;
uniform float uGamma;

void main() {
    vec4 hdrColor = texture(uSceneTexture, vTexCoord);
    
    // Apply tonemapping (would use functions from filters/tonemap.glsl)
    vec3 mapped = vec3(1.0) - exp(-hdrColor.rgb * uExposure);
    
    // Gamma correction
    mapped = pow(mapped, vec3(1.0 / uGamma));
    
    fragColor = vec4(mapped, hdrColor.a);
}