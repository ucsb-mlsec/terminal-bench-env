// Shadow mapping implementation for dynamic shadow generation
// This shader provides shadow map creation and sampling functionality

#include "common/utils.glsl"
#include "lighting/phong.glsl"

uniform sampler2D shadowMap;
uniform mat4 lightSpaceMatrix;
uniform vec3 lightPosition;
uniform float shadowBias;

// Calculate shadow map coordinates from world position
vec4 getShadowCoords(vec3 worldPos) {
    vec4 lightSpacePos = lightSpaceMatrix * vec4(worldPos, 1.0);
    lightSpacePos.xyz /= lightSpacePos.w;
    lightSpacePos.xyz = lightSpacePos.xyz * 0.5 + 0.5;
    return lightSpacePos;
}

// Sample shadow map with PCF filtering
float calculateShadow(vec3 worldPos, vec3 normal) {
    vec4 shadowCoords = getShadowCoords(worldPos);
    
    if(shadowCoords.z > 1.0) return 0.0;
    
    float currentDepth = shadowCoords.z;
    float bias = max(shadowBias * (1.0 - dot(normal, normalize(lightPosition))), shadowBias * 0.1);
    
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(shadowMap, shadowCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}