#version 450 core

out vec4 FragColor;

// Material calculation subroutines
subroutine vec3 MaterialFunc();
subroutine uniform MaterialFunc material;

subroutine(MaterialFunc) vec3 diffuse() {
    return vec3(0.8, 0.2, 0.2);
}

subroutine(MaterialFunc) vec3 specular() {
    return vec3(0.9, 0.9, 0.9);
}

subroutine(MaterialFunc) vec3 metallic() {
    return vec3(0.7, 0.7, 0.8);
}

subroutine(MaterialFunc) vec3 plastic() {
    return vec3(0.5, 0.5, 0.6);
}

// Lighting calculation subroutines
subroutine vec4 LightingFunc();
subroutine uniform LightingFunc lighting;

subroutine(LightingFunc) vec4 phong() {
    return vec4(1.0, 1.0, 1.0, 1.0);
}

subroutine(LightingFunc) vec4 blinn() {
    return vec4(0.95, 0.95, 0.95, 1.0);
}

subroutine(LightingFunc) vec4 toon() {
    return vec4(0.85, 0.85, 0.85, 1.0);
}

subroutine(LightingFunc) vec4 oren_nayar() {
    return vec4(0.9, 0.9, 0.9, 1.0);
}

// Texture sampling subroutines
subroutine float TextureFunc();
subroutine uniform TextureFunc textureSampler;

subroutine(TextureFunc) float standard() {
    return 1.0;
}

subroutine(TextureFunc) float parallax() {
    return 0.95;
}

subroutine(TextureFunc) float triplanar() {
    return 0.9;
}

// Shadow calculation subroutines
subroutine float ShadowFunc();
subroutine uniform ShadowFunc shadow;

subroutine(ShadowFunc) float hard_shadow() {
    return 0.0;
}

subroutine(ShadowFunc) float soft_shadow() {
    return 0.5;
}

subroutine(ShadowFunc) float pcf_shadow() {
    return 0.3;
}

subroutine(ShadowFunc) float no_shadow() {
    return 1.0;
}

void main() {
    vec3 materialColor = material();
    vec4 lightingResult = lighting();
    float textureFactor = textureSampler();
    float shadowFactor = shadow();
    
    vec3 finalColor = materialColor * lightingResult.rgb * textureFactor * shadowFactor;
    FragColor = vec4(finalColor, lightingResult.a);
}