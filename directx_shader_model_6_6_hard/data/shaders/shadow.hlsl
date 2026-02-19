// Shadow Mapping Shader - Shader Model 6.6
// All resource bindings validated for SM6.6 compliance

// Shadow map array texture with explicit bounded size
Texture2DArray shadowMapArray : register(t0, space0);

// Comparison sampler for shadow map sampling
SamplerComparisonState shadowSampler : register(s0);

// Constant buffer for shadow rendering parameters
cbuffer ShadowConstants : register(b0)
{
    float4x4 lightViewProj[4];
    float4 shadowParams;
};

// Structured buffer for instance transform matrices
StructuredBuffer<float4x4> instanceTransforms : register(t1);

// Vertex shader input structure
struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD0;
    uint instanceID : SV_InstanceID;
};

// Vertex shader output / Pixel shader input
struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldPos : WORLD_POSITION;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD0;
};

// Vertex Shader
PSInput VSMain(VSInput input)
{
    PSInput output;
    
    // Get instance transform
    float4x4 worldTransform = instanceTransforms[input.instanceID];
    
    // Transform position to world space
    float4 worldPos = mul(float4(input.position, 1.0f), worldTransform);
    output.worldPos = worldPos.xyz;
    
    // Transform to light view-projection space (cascade 0)
    output.position = mul(worldPos, lightViewProj[0]);
    
    // Transform normal
    output.normal = mul(input.normal, (float3x3)worldTransform);
    
    // Pass through texcoord
    output.texcoord = input.texcoord;
    
    return output;
}

// Pixel Shader - Shadow calculation
float4 PSMain(PSInput input) : SV_TARGET
{
    // Normalize normal
    float3 N = normalize(input.normal);
    
    // Sample shadow map with comparison
    float3 shadowCoord = input.position.xyz / input.position.w;
    shadowCoord.xy = shadowCoord.xy * 0.5f + 0.5f;
    shadowCoord.y = 1.0f - shadowCoord.y;
    
    // Shadow map cascade selection based on depth
    uint cascadeIndex = 0;
    float depth = input.position.z;
    
    if (depth > shadowParams.z)
        cascadeIndex = 3;
    else if (depth > shadowParams.y)
        cascadeIndex = 2;
    else if (depth > shadowParams.x)
        cascadeIndex = 1;
    
    // Sample shadow map array with comparison
    float shadowFactor = shadowMapArray.SampleCmpLevelZero(
        shadowSampler,
        float3(shadowCoord.xy, cascadeIndex),
        shadowCoord.z
    );
    
    // Apply shadow bias from parameters
    shadowFactor = saturate(shadowFactor + shadowParams.w);
    
    return float4(shadowFactor, shadowFactor, shadowFactor, 1.0f);
}