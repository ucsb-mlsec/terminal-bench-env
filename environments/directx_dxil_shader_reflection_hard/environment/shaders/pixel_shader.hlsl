// Pixel Shader for PBR Surface Shading
// pixel_shader.hlsl

// Constant Buffer for Material Properties
cbuffer MaterialConstants : register(b0)
{
    float4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
};

// Constant Buffer for Lighting Data
cbuffer LightingData : register(b1)
{
    float3 lightDirection;
    float lightIntensity;
    float3 lightColor;
    float ambientStrength;
    float3 viewPosition;
    float padding;
};

// Constant Buffer for Scene Parameters
cbuffer SceneParams : register(b2)
{
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float time;
    float exposure;
    float gamma;
    float iblStrength;
};

// Texture Resources
Texture2D albedoTexture : register(t0);
Texture2D normalMap : register(t1);
Texture2D metallicRoughnessMap : register(t2);
TextureCube environmentMap : register(t3);

// Sampler States
SamplerState anisotropicSampler : register(s0);
SamplerComparisonState shadowSampler : register(s1);

// Structured Buffer for Dynamic Lights
struct LightData
{
    float3 position;
    float radius;
    float3 color;
    float intensity;
};

StructuredBuffer<LightData> lightBuffer : register(t4);

// Input structure from vertex shader
struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPosition : WORLD_POSITION;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float3 bitangent : BITANGENT;
    float2 texCoord : TEXCOORD0;
};

// Output structure
struct PSOutput
{
    float4 color : SV_Target;
};

// Constants for PBR calculations
static const float PI = 3.14159265359;

// Normal Distribution Function (GGX/Trowbridge-Reitz)
float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    
    return num / denom;
}

// Geometry Function (Schlick-GGX)
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return num / denom;
}

// Fresnel Function (Schlick approximation)
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Main pixel shader function
PSOutput main(PSInput input)
{
    PSOutput output;
    
    // Sample textures
    float4 albedo = albedoTexture.Sample(anisotropicSampler, input.texCoord) * baseColorFactor;
    float3 sampledNormal = normalMap.Sample(anisotropicSampler, input.texCoord).xyz;
    float2 metallicRoughness = metallicRoughnessMap.Sample(anisotropicSampler, input.texCoord).bg;
    
    float metallic = metallicRoughness.x * metallicFactor;
    float roughness = metallicRoughness.y * roughnessFactor;
    
    // Transform normal from tangent space to world space
    sampledNormal = sampledNormal * 2.0 - 1.0;
    sampledNormal.xy *= normalScale;
    float3x3 TBN = float3x3(normalize(input.tangent), normalize(input.bitangent), normalize(input.normal));
    float3 N = normalize(mul(sampledNormal, TBN));
    
    // Calculate view direction
    float3 V = normalize(viewPosition - input.worldPosition);
    
    // Calculate base reflectivity (F0)
    float3 F0 = float3(0.04, 0.04, 0.04);
    F0 = lerp(F0, albedo.rgb, metallic);
    
    // Initialize radiance
    float3 Lo = float3(0.0, 0.0, 0.0);
    
    // Main directional light
    float3 L = normalize(-lightDirection);
    float3 H = normalize(V + L);
    float3 radiance = lightColor * lightIntensity;
    
    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySchlickGGX(max(dot(N, V), 0.0), roughness) * 
              GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
    float3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    
    float3 kS = F;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;
    kD *= 1.0 - metallic;
    
    float3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
    float3 specular = numerator / denominator;
    
    float NdotL = max(dot(N, L), 0.0);
    Lo += (kD * albedo.rgb / PI + specular) * radiance * NdotL;
    
    // Add point lights from structured buffer
    for (uint i = 0; i < 4; i++)
    {
        LightData light = lightBuffer[i];
        float3 lightVec = light.position - input.worldPosition;
        float distance = length(lightVec);
        float attenuation = 1.0 / (distance * distance);
        
        if (distance < light.radius)
        {
            float3 L_point = normalize(lightVec);
            float3 H_point = normalize(V + L_point);
            float3 radiance_point = light.color * light.intensity * attenuation;
            
            float NdotL_point = max(dot(N, L_point), 0.0);
            Lo += (kD * albedo.rgb / PI) * radiance_point * NdotL_point;
        }
    }
    
    // Sample environment map for ambient IBL
    float3 R = reflect(-V, N);
    float3 envColor = environmentMap.Sample(anisotropicSampler, R).rgb;
    float3 ambient = envColor * albedo.rgb * ambientStrength * iblStrength;
    
    float3 color = ambient + Lo;
    
    // Tone mapping and gamma correction
    color = color / (color + float3(1.0, 1.0, 1.0));
    color = pow(color, float3(1.0 / gamma, 1.0 / gamma, 1.0 / gamma));
    
    output.color = float4(color, albedo.a);
    
    return output;
}