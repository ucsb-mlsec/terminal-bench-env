cbuffer PostProcessParams : register(b0)
{
    float bloomThreshold;
    float bloomIntensity;
    float2 texelSize;
    float3 colorTint;
    float saturation;
    float contrast;
};

cbuffer VignetteSettings : register(b1)
{
    float vignetteIntensity;
    float vignetteSmoothness;
    float2 vignetteCenter;
    float3 vignetteColor;
    float vignetteRoundness;
};

cbuffer ChromaticAberration : register(b2)
{
    float aberrationStrength;
    float2 aberrationOffset;
    float distortionScale;
    float4 channelMultipliers;
    float focusDistance;
};