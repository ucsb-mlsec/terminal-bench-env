// Vertex Shader for Mesh Rendering
// Resource declarations for graphics pipeline integration

// Constant Buffer 1: Transformation matrices
cbuffer TransformData : register(b0)
{
    float4x4 worldMatrix;
    float4x4 viewMatrix;
    float4x4 projectionMatrix;
    float4x4 worldViewProjection;
};

// Constant Buffer 2: Material properties
cbuffer MaterialProperties : register(b1)
{
    float4 materialColor;
    float roughness;
    float metallic;
    float ambientOcclusion;
    float padding;
};

// Texture resource: Height map for displacement
Texture2D heightMap : register(t0);

// Texture resource: Normal map for detail
Texture2D normalMap : register(t1);

// Sampler for texture sampling
SamplerState linearSampler : register(s0);

// Structured buffer for per-instance data
StructuredBuffer<float4x4> instanceTransforms : register(t2);

// Input structure from vertex buffer
struct VertexInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD0;
    float3 tangent : TANGENT;
    uint instanceID : SV_InstanceID;
};

// Output structure to pixel shader
struct VertexOutput
{
    float4 position : SV_Position;
    float3 worldPosition : WORLD_POSITION;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD0;
    float3 tangent : TANGENT;
    float4 color : COLOR;
};

// Main vertex shader function
VertexOutput main(VertexInput input)
{
    VertexOutput output;
    
    // Sample height map for displacement
    float height = heightMap.SampleLevel(linearSampler, input.texcoord, 0).r;
    
    // Apply displacement along normal
    float3 displacedPosition = input.position + input.normal * height * 0.1f;
    
    // Get instance transform
    float4x4 instanceTransform = instanceTransforms[input.instanceID];
    
    // Transform vertex position to world space
    float4 worldPos = mul(float4(displacedPosition, 1.0f), instanceTransform);
    worldPos = mul(worldPos, worldMatrix);
    output.worldPosition = worldPos.xyz;
    
    // Transform to clip space
    output.position = mul(worldPos, viewMatrix);
    output.position = mul(output.position, projectionMatrix);
    
    // Transform normal to world space
    float3x3 normalMatrix = (float3x3)worldMatrix;
    output.normal = normalize(mul(input.normal, normalMatrix));
    
    // Transform tangent to world space
    output.tangent = normalize(mul(input.tangent, normalMatrix));
    
    // Pass through texture coordinates
    output.texcoord = input.texcoord;
    
    // Apply material color
    output.color = materialColor;
    
    return output;
}