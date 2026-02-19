// Skybox Shader - Shader Model 6.6
// Valid resource bindings for skybox rendering

// Constant buffer for transformation and exposure
cbuffer SkyboxConstants : register(b0)
{
    float4x4 viewProjection;
    float exposure;
    float3 padding;
};

// Skybox cubemap texture
TextureCube skyboxTexture : register(t0);

// Sampler for the skybox
SamplerState skyboxSampler : register(s0);

// Vertex shader input structure
struct VSInput
{
    float3 position : POSITION;
};

// Vertex shader output / Pixel shader input
struct PSInput
{
    float4 position : SV_POSITION;
    float3 texCoord : TEXCOORD0;
};

// Vertex Shader
PSInput VSMain(VSInput input)
{
    PSInput output;
    
    // Remove translation from view matrix by zeroing out the last column
    float4x4 viewNoTranslation = viewProjection;
    viewNoTranslation[3][0] = 0.0f;
    viewNoTranslation[3][1] = 0.0f;
    viewNoTranslation[3][2] = 0.0f;
    
    // Transform position to clip space
    output.position = mul(float4(input.position, 1.0f), viewNoTranslation);
    
    // Set z to w to ensure skybox is always at far plane
    output.position.z = output.position.w;
    
    // Use local position as texture coordinate
    output.texCoord = input.position;
    
    return output;
}

// Pixel Shader
float4 PSMain(PSInput input) : SV_TARGET
{
    // Sample the skybox cubemap
    float3 color = skyboxTexture.Sample(skyboxSampler, input.texCoord).rgb;
    
    // Apply exposure adjustment
    color *= exposure;
    
    return float4(color, 1.0f);
}