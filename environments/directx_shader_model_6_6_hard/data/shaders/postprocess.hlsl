// Shader Model 6.6 Post-Processing Shader
// All resource bindings are correctly declared following SM6.6 specifications

// Input textures - using 't' registers for SRVs
Texture2D sceneTexture : register(t0, space0);
Texture2D depthTexture : register(t1, space0);

// Samplers - using 's' registers
SamplerState pointSampler : register(s0);
SamplerState linearSampler : register(s1);

// Output texture - using 'u' register for UAV
RWTexture2D<float4> outputTexture : register(u0);

// Constant buffer for tone mapping parameters
cbuffer ToneMappingParams : register(b0)
{
    float exposure;
    float gamma;
    float contrast;
    float saturation;
};

// ACES Filmic Tone Mapping curve approximation
float3 ACESFilm(float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Convert RGB to luminance
float Luminance(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

// Adjust saturation
float3 AdjustSaturation(float3 color, float saturation)
{
    float lum = Luminance(color);
    return lerp(float3(lum, lum, lum), color, saturation);
}

// Main compute shader for tone mapping
[numthreads(8, 8, 1)]
void ToneMappingCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadID.xy;
    
    // Get output texture dimensions
    uint width, height;
    outputTexture.GetDimensions(width, height);
    
    // Early exit if outside bounds
    if (pixelCoord.x >= width || pixelCoord.y >= height)
        return;
    
    // Calculate UV coordinates
    float2 uv = (float2(pixelCoord) + 0.5f) / float2(width, height);
    
    // Sample scene texture using linear sampler
    float3 sceneColor = sceneTexture.SampleLevel(linearSampler, uv, 0).rgb;
    
    // Sample depth for potential depth-based effects
    float depth = depthTexture.SampleLevel(pointSampler, uv, 0).r;
    
    // Apply exposure
    float3 color = sceneColor * exposure;
    
    // Apply contrast adjustment
    color = pow(abs(color), float3(contrast, contrast, contrast)) * sign(color);
    
    // Apply saturation adjustment
    color = AdjustSaturation(color, saturation);
    
    // Apply ACES filmic tone mapping
    color = ACESFilm(color);
    
    // Apply gamma correction
    color = pow(abs(color), float3(1.0f / gamma, 1.0f / gamma, 1.0f / gamma));
    
    // Write final color to output
    outputTexture[pixelCoord] = float4(color, 1.0f);
}

// Alternative simple tone mapping compute shader
[numthreads(16, 16, 1)]
void SimpleToneMappingCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadID.xy;
    
    uint width, height;
    outputTexture.GetDimensions(width, height);
    
    if (pixelCoord.x >= width || pixelCoord.y >= height)
        return;
    
    float2 uv = (float2(pixelCoord) + 0.5f) / float2(width, height);
    
    // Sample scene with point sampling for performance
    float3 sceneColor = sceneTexture.SampleLevel(pointSampler, uv, 0).rgb;
    
    // Simple Reinhard tone mapping
    float3 mapped = sceneColor / (sceneColor + float3(1.0f, 1.0f, 1.0f));
    
    // Gamma correction
    mapped = pow(mapped, float3(1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f));
    
    outputTexture[pixelCoord] = float4(mapped, 1.0f);
}