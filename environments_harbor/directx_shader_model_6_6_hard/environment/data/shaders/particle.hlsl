// Particle System Shader - Shader Model 6.6
// This shader handles particle rendering with GPU-based simulation

struct ParticleData
{
    float3 position;
    float3 velocity;
    float4 color;
    float lifetime;
    float size;
};

// BINDING DECLARATIONS

// Particle data buffer - INVALID: negative space index
StructuredBuffer<ParticleData> particles : register(t0, space-1);

// Effect textures for particle rendering - INVALID: unbounded array with register
// This should be: Texture2D effects[] without explicit register or proper bounded syntax
Texture2D effects[] : register(t1, space0);

// Particle state output buffer - INVALID: RW resource should use 'u' register, not 't'
RWBuffer<float4> particleBuffer : register(t5);

// Valid sampler for effect textures
SamplerState linearSampler : register(s0);

// Camera constants - valid binding
cbuffer CameraConstants : register(b0)
{
    float4x4 viewProjection;
    float3 cameraPosition;
    float deltaTime;
};

// Particle rendering constants
cbuffer ParticleConstants : register(b1, space0)
{
    float4 globalTint;
    float fadeDistance;
    float maxParticles;
    float2 padding;
};

// Vertex shader output structure
struct VSOutput
{
    float4 position : SV_POSITION;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD0;
    float size : PSIZE;
};

// Simple particle vertex shader
VSOutput ParticleVS(uint vertexID : SV_VertexID)
{
    VSOutput output;
    
    uint particleIndex = vertexID / 4;
    uint cornerIndex = vertexID % 4;
    
    ParticleData particle = particles[particleIndex];
    
    // Calculate billboard corners
    float2 corners[4] = {
        float2(-1, -1),
        float2(1, -1),
        float2(-1, 1),
        float2(1, 1)
    };
    
    float2 corner = corners[cornerIndex] * particle.size;
    float3 worldPos = particle.position + float3(corner, 0);
    
    output.position = mul(float4(worldPos, 1.0), viewProjection);
    output.color = particle.color * globalTint;
    output.texcoord = corners[cornerIndex] * 0.5 + 0.5;
    output.size = particle.size;
    
    return output;
}

// Simple particle pixel shader
float4 ParticlePS(VSOutput input) : SV_TARGET
{
    // Sample from first effect texture
    float4 texColor = effects[0].Sample(linearSampler, input.texcoord);
    
    float4 finalColor = texColor * input.color;
    
    // Distance fade
    float distToCam = length(input.position.xyz - cameraPosition);
    float fade = saturate(1.0 - distToCam / fadeDistance);
    finalColor.a *= fade;
    
    return finalColor;
}