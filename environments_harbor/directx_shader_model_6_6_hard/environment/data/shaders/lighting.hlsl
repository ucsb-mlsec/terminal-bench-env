// Lighting Shader - Shader Model 6.6
// DirectX Lighting Pipeline Implementation

// INVALID: Unbounded array with incorrect bracket syntax - should be textures[] or textures[N]
Texture2D lightTextures[];

// INVALID: Texture using wrong register type - textures should use 't' register, not 'b'
Texture2D diffuseMap : register(b5);

// INVALID: SamplerState with invalid register - samplers should use 's' register, not 't'
SamplerState sampler : register(t10);

// Valid constant buffer for contrast
cbuffer LightData : register(b0)
{
    float4 lightColor;
    float4 ambientColor;
    float3 lightDirection;
    float lightIntensity;
};

// Additional resource with another error for good measure
// INVALID: Buffer using texture register
Buffer<float4> lightPositions : register(t2);

struct VSInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
    float2 texCoord : TEXCOORD;
    float3 worldPos : WORLDPOS;
};

float4 CalculateLighting(float3 normal, float3 worldPos, float2 texCoord)
{
    // Sample diffuse texture
    float4 diffuse = diffuseMap.Sample(sampler, texCoord);
    
    // Calculate basic Lambertian lighting
    float3 N = normalize(normal);
    float3 L = normalize(-lightDirection);
    float NdotL = saturate(dot(N, L));
    
    // Apply lighting
    float4 finalColor = diffuse * lightColor * NdotL * lightIntensity;
    finalColor += ambientColor * diffuse;
    
    return saturate(finalColor);
}

PSInput VSMain(VSInput input)
{
    PSInput output;
    output.position = float4(input.position, 1.0f);
    output.normal = input.normal;
    output.texCoord = input.texCoord;
    output.worldPos = input.position;
    return output;
}

float4 PSMain(PSInput input) : SV_TARGET
{
    return CalculateLighting(input.normal, input.worldPos, input.texCoord);
}