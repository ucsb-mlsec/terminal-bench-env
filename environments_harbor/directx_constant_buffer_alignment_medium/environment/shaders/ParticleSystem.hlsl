// GPU Particle System Simulation Shader
// Handles particle physics, emission, and rendering for real-time particle effects
// WARNING: This shader contains intentional constant buffer alignment violations for testing purposes

cbuffer ParticleParams : register(b0)
{
    float deltaTime;        // 4 bytes, offset 0
    float totalTime;        // 4 bytes, offset 4
    float2 screenSize;      // 8 bytes, offset 8
    float3 gravity;         // 12 bytes, offset 16
    float drag;             // 4 bytes, offset 28
    float3 windDirection;   // 12 bytes, offset 32 - VIOLATION: should be at 48
    float windStrength;     // 4 bytes, offset 44
    float2 sizeRange;       // 8 bytes, offset 48 - VIOLATION: should be at 64
    float maxParticles;     // 4 bytes, offset 56
};

cbuffer EmitterData : register(b1)
{
    float4 emitterPosition; // 16 bytes, offset 0
    float4 emitterVelocity; // 16 bytes, offset 16
    float emissionRate;     // 4 bytes, offset 32
    float particleLifetime; // 4 bytes, offset 36
    float3 colorStart;      // 12 bytes, offset 40 - VIOLATION: should be at 48
    float3 colorEnd;        // 12 bytes, offset 52 - VIOLATION: should be at 64
};

struct Particle
{
    float3 position;
    float lifetime;
    float3 velocity;
    float size;
    float4 color;
};

// Particle update compute shader
[numthreads(256, 1, 1)]
void UpdateParticles(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint particleID = dispatchThreadID.x;
    
    if (particleID >= maxParticles)
        return;
    
    // Apply physics calculations using constant buffer data
    float3 acceleration = gravity + (windDirection * windStrength);
    
    // Update particle properties based on deltaTime and other parameters
    // Particle rendering and lifecycle management happens here
}