// BasicLighting.hlsl
// Shader constant buffers for scene lighting and camera transformations
// Used in forward rendering pipeline for PBR lighting calculations

cbuffer SceneLighting : register(b0)
{
    float3 lightDirection;  // 12 bytes, offset 0
    float lightIntensity;   // 4 bytes, offset 12
    float3 ambientColor;    // 12 bytes, offset 16
    float specularPower;    // 4 bytes, offset 28
    float2 shadowBias;      // 8 bytes, offset 32 - VIOLATION: should be at 48
    float metallic;         // 4 bytes, offset 40
};

cbuffer CameraData : register(b1)
{
    float4x4 viewMatrix;        // 64 bytes, offset 0
    float4x4 projectionMatrix;  // 64 bytes, offset 64
    float3 cameraPosition;      // 12 bytes, offset 128
    float nearPlane;            // 4 bytes, offset 140
    float farPlane;             // 4 bytes, offset 144
};

// Vertex shader input structure
struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD0;
};

// Pixel shader input structure
struct PSInput
{
    float4 position : SV_POSITION;
    float3 worldNormal : NORMAL;
    float2 texCoord : TEXCOORD0;
    float3 worldPos : TEXCOORD1;
};

// Main vertex shader for lighting calculations
PSInput VSMain(VSInput input)
{
    PSInput output;
    output.position = mul(float4(input.position, 1.0f), viewMatrix);
    output.position = mul(output.position, projectionMatrix);
    output.worldNormal = input.normal;
    output.texCoord = input.texCoord;
    output.worldPos = input.position;
    return output;
}

// Main pixel shader for PBR lighting
float4 PSMain(PSInput input) : SV_TARGET
{
    float3 N = normalize(input.worldNormal);
    float3 L = normalize(-lightDirection);
    float3 V = normalize(cameraPosition - input.worldPos);
    
    float NdotL = max(dot(N, L), 0.0f);
    float3 diffuse = NdotL * lightIntensity;
    
    float3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0f);
    float3 specular = pow(NdotH, specularPower) * metallic;
    
    float3 finalColor = ambientColor + diffuse + specular;
    return float4(finalColor, 1.0f);
}