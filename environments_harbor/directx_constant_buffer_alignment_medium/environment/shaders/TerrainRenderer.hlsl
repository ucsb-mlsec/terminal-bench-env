// Terrain LOD Rendering Shader
// Handles dynamic level-of-detail terrain rendering with tessellation

cbuffer TerrainData : register(b0)
{
    float4x4 worldMatrix;  // 64 bytes, offset 0
    float3 terrainOffset;  // 12 bytes, offset 64
    float heightScale;     // 4 bytes, offset 76
    float2 terrainSize;    // 8 bytes, offset 80
    float tessellation;    // 4 bytes, offset 88
    float3 clipPlane;      // 12 bytes, offset 92
    int layerCount;        // 4 bytes, offset 104
};

cbuffer LODSettings : register(b1)
{
    float4 lodDistances;   // 16 bytes, offset 0
    float minTessellation; // 4 bytes, offset 16
    float maxTessellation; // 4 bytes, offset 20
    float2 patchSize;      // 8 bytes, offset 24
};

struct VS_INPUT
{
    float3 position : POSITION;
    float2 texCoord : TEXCOORD0;
};

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD0;
    float3 worldPos : TEXCOORD1;
};

VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    
    float3 worldPosition = input.position + terrainOffset;
    worldPosition.y *= heightScale;
    
    output.position = mul(float4(worldPosition, 1.0f), worldMatrix);
    output.texCoord = input.texCoord;
    output.worldPos = worldPosition;
    
    return output;
}