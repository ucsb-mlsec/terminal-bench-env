#version 450

// Input attributes from vertex shader
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec3 fragWorldPos;
in vec3 fragTangent;

// G-buffer outputs (multiple render targets)
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outPosition;
layout(location = 3) out vec4 outMaterial;

// Texture samplers
uniform sampler2D albedoMap;
uniform sampler2D normalMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

// Uniform parameters
uniform float normalStrength;
uniform float mipmapBias;

void main()
{
    // VRS ISSUE #1: Using derivatives to compute tangent-space basis
    // This breaks when fragments are coalesced in 2x2 or larger blocks
    vec3 pos_dx = dFdx(fragWorldPos);
    vec3 pos_dy = dFdy(fragWorldPos);
    vec2 texCoord_dx = dFdx(fragTexCoord);
    vec2 texCoord_dy = dFdy(fragTexCoord);
    
    // Compute geometric normal from position derivatives
    vec3 geometricNormal = normalize(cross(pos_dx, pos_dy));
    
    // VRS ISSUE #2: Compute tangent and bitangent from derivatives
    // This assumes per-pixel derivatives which VRS violates
    vec3 T = normalize(pos_dx * texCoord_dy.t - pos_dy * texCoord_dx.t);
    vec3 B = normalize(pos_dy * texCoord_dx.s - pos_dx * texCoord_dy.s);
    vec3 N = geometricNormal;
    mat3 TBN = mat3(T, B, N);
    
    // VRS ISSUE #3: Using fwidth for adaptive mipmap bias
    // fwidth returns incorrect values with coalesced fragments
    vec2 texCoordDerivMagnitude = fwidth(fragTexCoord);
    float derivativeBias = log2(max(texCoordDerivMagnitude.x, texCoordDerivMagnitude.y) * 2048.0);
    float adaptiveMipBias = mipmapBias + clamp(derivativeBias, -2.0, 2.0);
    
    // Sample albedo with computed mipmap bias
    vec4 albedo = texture(albedoMap, fragTexCoord, adaptiveMipBias);
    
    // Sample normal map
    vec3 tangentNormal = texture(normalMap, fragTexCoord).xyz * 2.0 - 1.0;
    tangentNormal.xy *= normalStrength;
    
    // VRS ISSUE #4: Transform normal using derivative-computed TBN matrix
    // The TBN matrix is incorrect with VRS, leading to wrong normals
    vec3 worldNormal = normalize(TBN * tangentNormal);
    
    // Detect if we're on an edge using derivative magnitude
    // VRS ISSUE #5: Edge detection using derivatives
    float edgeFactor = length(fwidth(fragWorldPos));
    float isEdge = step(0.05, edgeFactor);
    
    // Adjust normal strength based on edge detection
    worldNormal = mix(worldNormal, geometricNormal, isEdge * 0.3);
    
    // Calculate anisotropic filtering compensation
    // VRS ISSUE #6: Manual anisotropy calculation with derivatives
    vec2 dx = dFdx(fragTexCoord * 2048.0);
    vec2 dy = dFdy(fragTexCoord * 2048.0);
    float anisoRatio = length(dx) / (length(dy) + 0.001);
    float anisoCorrection = clamp(anisoRatio, 0.5, 2.0);
    
    // Sample roughness and AO with anisotropic correction
    float roughness = texture(roughnessMap, fragTexCoord).r * anisoCorrection;
    float ao = texture(aoMap, fragTexCoord).r;
    
    // Compute screen-space derivatives for position precision
    vec3 positionGradient = abs(dFdx(fragWorldPos)) + abs(dFdy(fragWorldPos));
    float positionComplexity = dot(positionGradient, vec3(1.0));
    
    // Adjust metallic based on geometric complexity
    float metallic = clamp(positionComplexity * 0.1, 0.0, 1.0);
    
    // Enhanced normal with derivative-based detail
    vec3 normalDetail = vec3(
        dFdx(fragNormal.y) - dFdy(fragNormal.x),
        dFdy(fragNormal.z) - dFdx(fragNormal.y),
        dFdx(fragNormal.x) - dFdy(fragNormal.z)
    );
    worldNormal += normalDetail * 0.1;
    worldNormal = normalize(worldNormal);
    
    // Output to G-buffer targets
    outAlbedo = vec4(albedo.rgb, 1.0);
    outNormal = vec4(worldNormal * 0.5 + 0.5, 1.0);
    outPosition = vec4(fragWorldPos, positionComplexity);
    outMaterial = vec4(roughness, metallic, ao, 1.0);
    
    // Additional derivative-based ambient term
    float ambientOcclusion = ao * (1.0 - edgeFactor * 0.2);
    outMaterial.b = ambientOcclusion;
}
```

Now I'll analyze this shader for VRS compatibility issues and create the analysis.json file:

```json
{
  "vrs_compatible": false,
  "issues_found": 6,
  "max_safe_rate": "1x1"
}