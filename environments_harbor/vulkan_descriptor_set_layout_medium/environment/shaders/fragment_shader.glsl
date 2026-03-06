#version 450

// Descriptor bindings for fragment shader
layout(set = 0, binding = 0) uniform sampler2D diffuseTexture;
layout(set = 0, binding = 1) uniform sampler2D normalMap;
layout(set = 0, binding = 2) readonly buffer MaterialBuffer {
    vec4 ambientColor;
    vec4 diffuseColor;
    vec4 specularColor;
    float shininess;
    float opacity;
    int useNormalMap;
    int useSpecularMap;
} materialData;
layout(set = 0, binding = 3) uniform sampler2D specularMap;

// Input variables from vertex shader
layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragTangent;
layout(location = 4) in vec3 fragBitangent;

// Output color
layout(location = 0) out vec4 outColor;

// Push constants for lighting
layout(push_constant) uniform PushConstants {
    vec3 lightPosition;
    vec3 viewPosition;
} pushConstants;

vec3 calculateNormal() {
    if (materialData.useNormalMap != 0) {
        // Sample normal map
        vec3 tangentNormal = texture(normalMap, fragTexCoord).rgb * 2.0 - 1.0;
        
        // Create TBN matrix
        vec3 T = normalize(fragTangent);
        vec3 B = normalize(fragBitangent);
        vec3 N = normalize(fragNormal);
        mat3 TBN = mat3(T, B, N);
        
        return normalize(TBN * tangentNormal);
    } else {
        return normalize(fragNormal);
    }
}

void main() {
    // Sample diffuse texture
    vec4 diffuseSample = texture(diffuseTexture, fragTexCoord);
    
    // Calculate normal
    vec3 normal = calculateNormal();
    
    // Lighting calculations
    vec3 lightDir = normalize(pushConstants.lightPosition - fragPosition);
    vec3 viewDir = normalize(pushConstants.viewPosition - fragPosition);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    
    // Ambient component
    vec3 ambient = materialData.ambientColor.rgb * diffuseSample.rgb;
    
    // Diffuse component
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * materialData.diffuseColor.rgb * diffuseSample.rgb;
    
    // Specular component
    vec3 specular = vec3(0.0);
    if (materialData.useSpecularMap != 0) {
        float spec = pow(max(dot(normal, halfwayDir), 0.0), materialData.shininess);
        vec3 specularSample = texture(specularMap, fragTexCoord).rgb;
        specular = spec * materialData.specularColor.rgb * specularSample;
    } else {
        float spec = pow(max(dot(normal, halfwayDir), 0.0), materialData.shininess);
        specular = spec * materialData.specularColor.rgb;
    }
    
    // Combine lighting components
    vec3 finalColor = ambient + diffuse + specular;
    
    // Apply opacity from material buffer
    outColor = vec4(finalColor, diffuseSample.a * materialData.opacity);
}