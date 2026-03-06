// Broken Pixel Shader - Contains multiple compilation errors
// This shader is intended to sample a texture and perform basic lighting

// Texture and sampler declarations
Texture diffuseTexture;
sampler diffuseSampler;

// Input structure from vertex shader
struct PS_INPUT
{
    Float4 position : SV_POSITION;
    float2 texCoord : TEXTURE0;
    float3 normal : NORM;
};

// Main pixel shader function
vector4 main(PS_INPUT input)
{
    // Sample the diffuse texture
    Float4 texColor = diffuseTexture.Sample(diffuseSampler, input.texCoord);
    
    // Simple directional lighting
    float3 lightDir = normalize(float3(0.5, -0.7, 0.3));
    float3 normal = normalize(input.normal);
    
    // Calculate diffuse lighting term
    float diffuse = saturate(dot(normal, -lightDir));
    
    // Combine texture color with lighting
    Float4 finalColor;
    finalColor.rgb = texColor.rgb * diffuse;
    finalColor.a = texColor.a;
    
    // Missing return statement and semantic on return type
}