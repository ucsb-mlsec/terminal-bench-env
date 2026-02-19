// PostProcess.hlsl
// Handles post-processing effects including bloom and color grading

cbuffer BloomSettings : register(b0)
{
    float threshold;     // 4 bytes, offset 0
    float intensity;     // 4 bytes, offset 4
    float radius;        // 4 bytes, offset 8
    float3 tint;         // 12 bytes, offset 12 - VIOLATION: should be at 16
    int iterations;      // 4 bytes, offset 24
};

cbuffer ColorGrading : register(b1)
{
    float4 saturation;   // 16 bytes, offset 0
    float4 contrast;     // 16 bytes, offset 16
    float4 gamma;        // 16 bytes, offset 32
    float exposure;      // 4 bytes, offset 48
    float temperature;   // 4 bytes, offset 52
    float vignette;      // 4 bytes, offset 56
};