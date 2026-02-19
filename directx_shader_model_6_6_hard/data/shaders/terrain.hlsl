// Terrain Rendering Shader - Shader Model 6.6
// Valid resource binding declarations

// Unbounded texture array for terrain materials
Texture2D textures[] : register(t0, space0);

// Linear sampler for texture filtering
SamplerState samplerLinear : register(s0);

// Scene constant buffer
cbuffer SceneConstants : register(b0)
{
    float4x4 viewProjection;
    float3 cameraPos;
    float padding;
};

// Bounded shadow map array
Texture2D shadowMaps[4] : register(t1);

// Vertex shader output structure
struct VSOutput
{
    float4 position : SV_POSITION;
    float3 worldPos : WORLD_POS;
    float2 uv : TEXCOORD0;
    float3 normal : NORMAL;
    uint materialIndex : MATERIAL_ID;
};

// Pixel shader for terrain rendering
float4 PSMain(VSOutput input) : SV_TARGET
{
    // Sample base terrain texture using bindless indexing
    float4 baseColor = textures[input.materialIndex].Sample(samplerLinear, input.uv);
    
    // Calculate view direction
    float3 viewDir = normalize(cameraPos - input.worldPos);
    
    // Simple lighting calculation
    float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
    float ndotl = saturate(dot(input.normal, lightDir));
    
    // Sample first shadow map for basic shadowing
    float2 shadowUV = input.worldPos.xz * 0.01;
    float shadowTerm = shadowMaps[0].Sample(samplerLinear, shadowUV).r;
    
    // Combine lighting
    float3 finalColor = baseColor.rgb * ndotl * shadowTerm;
    
    // Add ambient term
    finalColor += baseColor.rgb * 0.2;
    
    return float4(finalColor, baseColor.a);
}

// Vertex shader for terrain
VSOutput VSMain(
    float3 position : POSITION,
    float3 normal : NORMAL,
    float2 uv : TEXCOORD0,
    uint materialIndex : MATERIAL_ID)
{
    VSOutput output;
    
    output.worldPos = position;
    output.position = mul(float4(position, 1.0), viewProjection);
    output.uv = uv;
    output.normal = normal;
    output.materialIndex = materialIndex;
    
    return output;
}