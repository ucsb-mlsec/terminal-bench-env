// PBR Material Rendering - Constant Buffer Definitions
// This shader defines material properties and texture indices for physically-based rendering

cbuffer MaterialParams : register(b0)
{
    float4 baseColor;      // 16 bytes, offset 0
    float roughness;       // 4 bytes, offset 16
    float metalness;       // 4 bytes, offset 20
    float3 emissive;       // 12 bytes, offset 24 - VIOLATION: should be at 32
    float2 uvScale;        // 8 bytes, offset 36 - VIOLATION: should be at 48
    float opacity;         // 4 bytes, offset 44
    float3 normalStrength; // 12 bytes, offset 48 - VIOLATION: should be at 64
};

cbuffer TextureIndices : register(b1)
{
    int albedoIndex;    // 4 bytes, offset 0
    int normalIndex;    // 4 bytes, offset 4
    int roughnessIndex; // 4 bytes, offset 8
    int metallicIndex;  // 4 bytes, offset 12
};