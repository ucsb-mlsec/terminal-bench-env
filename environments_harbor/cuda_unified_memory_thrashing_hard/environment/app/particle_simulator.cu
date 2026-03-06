#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_PARTICLES 100000
#define DT 0.01f
#define NUM_STEPS 1000
#define BLOCK_SIZE 256

// Particle structure for physics simulation
struct Particle {
    float3 position;
    float3 velocity;
    float mass;
    int id;
};

// Utility function to make float3
__host__ __device__ float3 make_float3_helper(float x, float y, float z) {
    float3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return result;
}

// Utility function for float3 addition
__host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3_helper(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Utility function for float3 scaling
__host__ __device__ float3 operator*(const float3& a, float scalar) {
    return make_float3_helper(a.x * scalar, a.y * scalar, a.z * scalar);
}

// Calculate distance between two particles
__device__ float distance(float3 a, float3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

// PATTERN 1: Unified memory allocation without any memory hints or prefetching
void allocateParticles(Particle** particles, int count) {
    // Allocate unified memory for particle array
    size_t size = count * sizeof(Particle);
    cudaMallocManaged(particles, size);
    
    // Initialize particles on host side
    for (int i = 0; i < count; i++) {
        (*particles)[i].position.x = (float)(rand() % 1000) - 500.0f;
        (*particles)[i].position.y = (float)(rand() % 1000) - 500.0f;
        (*particles)[i].position.z = (float)(rand() % 1000) - 500.0f;
        
        (*particles)[i].velocity.x = (float)(rand() % 100) / 100.0f - 0.5f;
        (*particles)[i].velocity.y = (float)(rand() % 100) / 100.0f - 0.5f;
        (*particles)[i].velocity.z = (float)(rand() % 100) / 100.0f - 0.5f;
        
        (*particles)[i].mass = 1.0f + (float)(rand() % 100) / 100.0f;
        (*particles)[i].id = i;
    }
    
    // No cudaMemAdvise calls
    // No cudaMemPrefetchAsync calls
    // No preferred location hints
    // This will cause page faults on first GPU access
}

// Apply boundary conditions to keep particles in simulation space
__host__ __device__ void applyBoundaries(Particle* p) {
    float boundary = 1000.0f;
    if (p->position.x > boundary) p->position.x = boundary;
    if (p->position.x < -boundary) p->position.x = -boundary;
    if (p->position.y > boundary) p->position.y = boundary;
    if (p->position.y < -boundary) p->position.y = -boundary;
    if (p->position.z > boundary) p->position.z = boundary;
    if (p->position.z < -boundary) p->position.z = -boundary;
}

// PATTERN 2: Non-coalesced memory access pattern causing excessive page faults
__global__ void computeForces(Particle* particles, int count, float3* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    float3 total_force = make_float3_helper(0.0f, 0.0f, 0.0f);
    
    // Scattered memory access pattern - accessing particles in non-sequential order
    // This causes poor memory coalescing and excessive page migrations
    for (int i = 0; i < count; i += 97) {  // Prime number stride for worst-case access
        int other_idx = (idx * 73 + i * 37) % count;  // Random-like access pattern
        
        if (other_idx != idx) {
            float3 pos1 = particles[idx].position;
            float3 pos2 = particles[other_idx].position;
            float dist = distance(pos1, pos2);
            
            if (dist > 0.1f && dist < 100.0f) {
                float force_magnitude = (particles[idx].mass * particles[other_idx].mass) / (dist * dist);
                float3 direction = make_float3_helper(
                    (pos2.x - pos1.x) / dist,
                    (pos2.y - pos1.y) / dist,
                    (pos2.z - pos1.z) / dist
                );
                total_force = total_force + (direction * force_magnitude);
            }
        }
    }
    
    forces[idx] = total_force;
}

// Update particle positions and velocities
__global__ void updateParticles(Particle* particles, float3* forces, int count, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Update velocity based on force
    float3 acceleration = forces[idx] * (1.0f / particles[idx].mass);
    particles[idx].velocity = particles[idx].velocity + (acceleration * dt);
    
    // Update position based on velocity
    particles[idx].position = particles[idx].position + (particles[idx].velocity * dt);
}

// PATTERN 3: Interleaved host-device access creating constant page migration
void runSimulation(Particle* particles, int count, int num_steps) {
    float3* forces;
    cudaMallocManaged(&forces, count * sizeof(float3));
    
    int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("Starting simulation with %d particles for %d steps...\n", count, num_steps);
    
    for (int step = 0; step < num_steps; step++) {
        // Launch kernel to compute forces on GPU
        computeForces<<<blocks, BLOCK_SIZE>>>(particles, count, forces);
        cudaDeviceSynchronize();
        
        // Immediately access particle data on host - causes page migration from device to host
        if (step % 10 == 0) {
            float avg_velocity = 0.0f;
            for (int i = 0; i < count; i += 100) {
                avg_velocity += sqrtf(particles[i].velocity.x * particles[i].velocity.x +
                                     particles[i].velocity.y * particles[i].velocity.y +
                                     particles[i].velocity.z * particles[i].velocity.z);
            }
            avg_velocity /= (count / 100);
        }
        
        // Launch another kernel - causes page migration back from host to device
        updateParticles<<<blocks, BLOCK_SIZE>>>(particles, forces, count, DT);
        cudaDeviceSynchronize();
        
        // More host-side access immediately after kernel - more thrashing
        for (int i = 0; i < count; i += 500) {
            applyBoundaries(&particles[i]);
            // Additional modification on host side
            if (particles[i].position.x > 900.0f) {
                particles[i].velocity.x *= -0.5f;
            }
        }
        
        // No prefetching between these accesses
        // No memory locality hints
        // Causes continuous page migration CPU <-> GPU
    }
    
    cudaFree(forces);
}

// Calculate system statistics on host
void calculateStatistics(Particle* particles, int count) {
    double total_kinetic_energy = 0.0;
    double avg_x = 0.0, avg_y = 0.0, avg_z = 0.0;
    
    // Host access to all particle data
    for (int i = 0; i < count; i++) {
        float vx = particles[i].velocity.x;
        float vy = particles[i].velocity.y;
        float vz = particles[i].velocity.z;
        float v_squared = vx * vx + vy * vy + vz * vz;
        total_kinetic_energy += 0.5 * particles[i].mass * v_squared;
        
        avg_x += particles[i].position.x;
        avg_y += particles[i].position.y;
        avg_z += particles[i].position.z;
    }
    
    avg_x /= count;
    avg_y /= count;
    avg_z /= count;
    
    printf("Total kinetic energy: %f\n", total_kinetic_energy);
    printf("Average position: (%f, %f, %f)\n", avg_x, avg_y, avg_z);
}

// Collision detection kernel with poor memory access patterns
__global__ void detectCollisions(Particle* particles, int count, int* collision_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    int local_collisions = 0;
    
    // Another scattered access pattern
    for (int i = 0; i < count; i += 53) {
        int other_idx = (idx + i) % count;
        if (other_idx != idx) {
            float dist = distance(particles[idx].position, particles[other_idx].position);
            if (dist < 2.0f) {
                local_collisions++;
            }
        }
    }
    
    if (local_collisions > 0) {
        atomicAdd(collision_count, local_collisions);
    }
}

int main() {
    printf("Particle Physics Simulator with Unified Memory\n");
    printf("===============================================\n");
    
    Particle* particles = NULL;
    
    // Allocate and initialize particles (PATTERN 1 - no memory hints)
    allocateParticles(&particles, NUM_PARTICLES);
    
    printf("Allocated %d particles in unified memory\n", NUM_PARTICLES);
    
    // Run main simulation (PATTERN 3 - interleaved access)
    runSimulation(particles, NUM_PARTICLES, NUM_STEPS);
    
    // Detect collisions with poor access pattern
    int* collision_count;
    cudaMallocManaged(&collision_count, sizeof(int));
    *collision_count = 0;
    
    int blocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    detectCollisions<<<blocks, BLOCK_SIZE>>>(particles, NUM_PARTICLES, collision_count);
    cudaDeviceSynchronize();
    
    printf("Detected %d potential collisions\n", *collision_count);
    
    // Calculate final statistics (host access after GPU work)
    calculateStatistics(particles, NUM_PARTICLES);
    
    // Cleanup
    cudaFree(collision_count);
    cudaFree(particles);
    
    printf("Simulation complete\n");
    
    return 0;
}