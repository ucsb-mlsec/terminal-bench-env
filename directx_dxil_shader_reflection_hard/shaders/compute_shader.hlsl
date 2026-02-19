// Post-Processing Compute Shader
// Performs bilateral blur and particle simulation

// Constant Buffers
cbuffer ComputeParams : register(b0)
{
    float4x4 viewProjection;
    float2 texelSize;
    float blurRadius;
    float sigmaColor;
    float sigmaSpatial;
    float deltaTime;
    uint particleCount;
    uint frameIndex;
};

cbuffer FilterSettings : register(b1)
{
    float4 tintColor;
    float brightness;
    float contrast;
    float saturation;
    float vignetteStrength;
    float2 vignetteCenter;
    float vignetteRadius;
    float padding;
};

// Texture Resources
Texture2D<float4> inputTexture : register(t0);
Texture2D<float4> referenceTexture : register(t1);
Texture2D<float> depthTexture : register(t2);

// Sampler
SamplerState pointSampler : register(s0);

// Structured Buffers
struct ParticleData
{
    float3 position;
    float lifetime;
    float3 velocity;
    float size;
    float4 color;
};

StructuredBuffer<ParticleData> inputParticles : register(t3);
RWStructuredBuffer<ParticleData> outputParticles : register(u0);
RWStructuredBuffer<uint> counterBuffer : register(u1);

// Read-Write Output Texture
RWTexture2D<float4> outputTexture : register(u2);

// Thread group size
[numthreads(8, 8, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixelCoord = dispatchThreadID.xy;
    uint particleIndex = dispatchThreadID.x;
    
    // Bilateral blur post-processing
    float4 centerColor = inputTexture.Load(int3(pixelCoord, 0));
    float centerDepth = depthTexture.Load(int3(pixelCoord, 0));
    
    float4 blurredColor = float4(0, 0, 0, 0);
    float weightSum = 0.0f;
    
    // Sample neighboring pixels
    for (int y = -2; y <= 2; y++)
    {
        for (int x = -2; x <= 2; x++)
        {
            int2 sampleCoord = int2(pixelCoord) + int2(x, y);
            float4 sampleColor = inputTexture.Load(int3(sampleCoord, 0));
            float sampleDepth = depthTexture.Load(int3(sampleCoord, 0));
            
            // Spatial weight
            float spatialDist = length(float2(x, y)) * blurRadius;
            float spatialWeight = exp(-spatialDist * spatialDist / (2.0f * sigmaSpatial * sigmaSpatial));
            
            // Color weight
            float colorDist = length(sampleColor.rgb - centerColor.rgb);
            float colorWeight = exp(-colorDist * colorDist / (2.0f * sigmaColor * sigmaColor));
            
            // Depth weight
            float depthDist = abs(sampleDepth - centerDepth);
            float depthWeight = exp(-depthDist * depthDist / 0.01f);
            
            float weight = spatialWeight * colorWeight * depthWeight;
            blurredColor += sampleColor * weight;
            weightSum += weight;
        }
    }
    
    if (weightSum > 0.0f)
    {
        blurredColor /= weightSum;
    }
    
    // Apply filter settings
    blurredColor.rgb *= brightness;
    blurredColor.rgb = lerp(float3(0.5f, 0.5f, 0.5f), blurredColor.rgb, contrast);
    
    // Apply vignette
    float2 uv = (float2(pixelCoord) + 0.5f) * texelSize;
    float vignetteAmount = length(uv - vignetteCenter) / vignetteRadius;
    vignetteAmount = saturate(vignetteAmount);
    vignetteAmount = pow(vignetteAmount, 2.0f);
    blurredColor.rgb *= lerp(1.0f, 0.0f, vignetteAmount * vignetteStrength);
    
    // Blend with tint color
    blurredColor.rgb = lerp(blurredColor.rgb, tintColor.rgb, tintColor.a);
    
    // Write output
    outputTexture[pixelCoord] = blurredColor;
    
    // Particle simulation
    if (particleIndex < particleCount)
    {
        ParticleData particle = inputParticles[particleIndex];
        
        // Update particle physics
        particle.velocity.y -= 9.81f * deltaTime;
        particle.position += particle.velocity * deltaTime;
        particle.lifetime -= deltaTime;
        
        // Check if particle is still alive
        if (particle.lifetime > 0.0f)
        {
            outputParticles[particleIndex] = particle;
            
            // Increment alive counter
            uint originalValue;
            InterlockedAdd(counterBuffer[0], 1, originalValue);
        }
        else
        {
            // Reset dead particle
            particle.lifetime = 0.0f;
            particle.position = float3(0, 0, 0);
            particle.velocity = float3(0, 0, 0);
            outputParticles[particleIndex] = particle;
        }
    }
}