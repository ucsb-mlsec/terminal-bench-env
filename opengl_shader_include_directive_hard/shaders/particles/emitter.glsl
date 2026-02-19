// Particle Emitter System
// Handles spawning and initialization of particles

#include "particles/particle.glsl"

// Emitter configuration
uniform vec3 emitterPosition;
uniform float emissionRate;
uniform float particleLifetime;

// Random number generator seed
float random(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// Emit a new particle at the emitter position
Particle emitParticle(float time, int particleId) {
    Particle p;
    vec2 seed = vec2(time, float(particleId));
    
    p.position = emitterPosition;
    p.velocity = vec3(
        random(seed) * 2.0 - 1.0,
        random(seed + vec2(1.0, 0.0)) * 2.0 - 1.0,
        random(seed + vec2(0.0, 1.0)) * 2.0 - 1.0
    );
    p.lifetime = particleLifetime;
    p.age = 0.0;
    
    return p;
}

// Calculate number of particles to spawn this frame
int getSpawnCount(float deltaTime) {
    return int(emissionRate * deltaTime);
}