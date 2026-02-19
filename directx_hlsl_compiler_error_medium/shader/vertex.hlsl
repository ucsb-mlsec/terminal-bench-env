cbuffer TransformBuffer : register(b0)
{
    float4x4 worldMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
};

struct VS_INPUT
{
    float3 position : POSITION;
    float2 texCoord : TEXCOORD0;
    float3 normal : NORMAL;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
    float3 normal : NORMAL;
    float3 worldPos : TEXCOORD1;
};

VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    
    // Transform position to world space
    float4 worldPosition = mul(float4(input.position, 1.0f), worldMatrix);
    output.worldPos = worldPosition.xyz;
    
    // Transform to view space
    float4 viewPosition = mul(worldPosition, viewMatrix);
    
    // Transform to projection space
    output.position = mul(viewPosition, projectionMatrix);
    
    // Pass through texture coordinates
    output.texCoord = input.texCoord;
    
    // Transform normal to world space
    output.normal = mul(input.normal, (float3x3)worldMatrix);
    output.normal = normalize(output.normal);
    
    return output;
}