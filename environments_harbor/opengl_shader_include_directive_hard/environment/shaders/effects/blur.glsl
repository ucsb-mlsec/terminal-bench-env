// Gaussian Blur Effect Shader
// Implements multi-pass Gaussian blur for post-processing effects

#include "effects/bloom.glsl"

uniform sampler2D inputTexture;
uniform vec2 resolution;
uniform float blurRadius;
uniform int blurSamples;

// Gaussian blur weights for a 9-tap kernel
const float gaussianWeights[9] = float[](
    0.0162162162, 0.0540540541, 0.1216216216, 0.1945945946,
    0.2270270270,
    0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162
);

// Horizontal blur pass
vec4 blurHorizontal(vec2 texCoord) {
    vec4 result = vec4(0.0);
    vec2 offset = vec2(1.0 / resolution.x, 0.0) * blurRadius;
    
    for (int i = -4; i <= 4; i++) {
        vec2 sampleCoord = texCoord + offset * float(i);
        result += texture(inputTexture, sampleCoord) * gaussianWeights[i + 4];
    }
    
    return result;
}

// Vertical blur pass
vec4 blurVertical(vec2 texCoord) {
    vec4 result = vec4(0.0);
    vec2 offset = vec2(0.0, 1.0 / resolution.y) * blurRadius;
    
    for (int i = -4; i <= 4; i++) {
        vec2 sampleCoord = texCoord + offset * float(i);
        result += texture(inputTexture, sampleCoord) * gaussianWeights[i + 4];
    }
    
    return result;
}