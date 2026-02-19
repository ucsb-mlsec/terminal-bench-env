cbuffer TransformBuffer : register(b0)
{
    float4x4 worldMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float3 cameraPosition;
    float time;
};

cbuffer InstanceData : register(b1)
{
    float4x4 instanceTransform;
    float3 instanceColor;
    float instanceScale;
    float2 uvOffset;
};

cbuffer VertexAnimationBuffer : register(b2)
{
    float3 windDirection;
    float windStrength;
    float3 waveParams;
    float animationTime;
    float2 scrollSpeed;
    float morphWeight;
};