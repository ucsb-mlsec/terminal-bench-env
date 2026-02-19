cbuffer LightingData : register(b0)
{
    float3 lightPosition;
    float lightIntensity;
    float3 lightColor;
    float specularPower;
    float3 ambientColor;
};

cbuffer MaterialProperties : register(b1)
{
    float4 albedoColor;
    float metallic;
    float roughness;
    float3 emissiveColor;
    float ambientOcclusion;
    float2 textureScale;
};

cbuffer ShadowParameters : register(b2)
{
    float4x4 shadowMatrix;
    float3 shadowBias;
    float shadowStrength;
    float2 shadowMapSize;
    float pcfRadius;
};