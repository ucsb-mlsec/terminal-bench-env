// Shader Model: cs_6_0
// Compute Kernel for Image Processing Pipeline
// Purpose: Applies a brightness adjustment to input texture

RWTexture2D<float4> OutputTexture : register(u0);
Texture2D<float4> InputTexture : register(t0);
RWBuffer<float> DebugBuffer : register(u1);

cbuffer ComputeParams : register(b0)
{
    float brightnessMultiplier;
    uint2 textureDimensions;
    float contrastFactor;
    uint frameIndex;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadIndex = dispatchThreadID.x;
    
    // Calculate 2D texture coordinates from 1D thread index
    uint2 texCoord;
    texCoord.x = threadIndex % textureDimensions.x;
    texCoord.y = threadIndex / textureDimensions.x;
    
    // Bounds check
    if (texCoord.x >= textureDimensions.x || texCoord.y >= textureDimensions.y)
        return;
    
    // Read input pixel
    float4 inputColor = InputTexture[texCoord];
    
    // Apply brightness and contrast adjustment
    float4 outputColor;
    outputColor.rgb = (inputColor.rgb - 0.5f) * contrastFactor + 0.5f;
    outputColor.rgb *= brightnessMultiplier;
    outputColor.a = inputColor.a;
    
    // Write to output texture
    OutputTexture[texCoord] = saturate(outputColor);
    
    // Debug: Store luminance value
    float luminance = dot(outputColor.rgb, float3(0.299f, 0.587f, 0.114f));
    DebugBuffer[threadIndex] = luminance;
}