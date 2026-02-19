#version 430 core

#include "particles/emitter.glsl"

layout(local_size_x = 256) in;

struct Particle {
    vec3 position;
    vec3 velocity;
    vec4 color;
    float lifetime;
    float size;
};

layout(std430, binding = 0) buffer ParticleBuffer {
    Particle particles[];
};

uniform float deltaTime;
uniform vec3 gravity;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    Particle p = particles[idx];
    
    if (p.lifetime > 0.0) {
        p.velocity += gravity * deltaTime;
        p.position += p.velocity * deltaTime;
        p.lifetime -= deltaTime;
        p.color.a = p.lifetime / getMaxLifetime();
        
        particles[idx] = p;
    } else {
        particles[idx] = emitNewParticle(idx);
    }
}