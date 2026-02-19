#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>

struct Particle {
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;
    double mass;
};

class ParticleSimulation {
private:
    Particle* particles;
    int N;
    double dt;
    double box_size;
    int* neighbor_list;
    int* neighbor_count;
    
public:
    ParticleSimulation(int num_particles);
    ~ParticleSimulation();
    void calculateForces();
    void integrate();
    void applyBoundaryConditions();
    double calculateEnergy();
    void updateNeighborList();
    void runSimulation(int steps);
    void initializeParticles();
};

ParticleSimulation::ParticleSimulation(int num_particles) : N(num_particles) {
    particles = new Particle[N];
    neighbor_list = new int[N * 50];
    neighbor_count = new int[N];
    dt = 0.001;
    box_size = 10.0;
    initializeParticles();
}

ParticleSimulation::~ParticleSimulation() {
    delete[] particles;
    delete[] neighbor_list;
    delete[] neighbor_count;
}

void ParticleSimulation::initializeParticles() {
    for (int i = 0; i < N; i++) {
        particles[i].x = (double)rand() / RAND_MAX * box_size;
        particles[i].y = (double)rand() / RAND_MAX * box_size;
        particles[i].z = (double)rand() / RAND_MAX * box_size;
        particles[i].vx = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        particles[i].vy = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        particles[i].vz = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
        particles[i].fz = 0.0;
        particles[i].mass = 1.0;
        neighbor_count[i] = 0;
    }
}

void ParticleSimulation::calculateForces() {
    Particle* p = particles;
    int n = N;
    
    #pragma omp target teams
    {
        for (int i = 0; i < n; i++) {
            p[i].fx = 0.0;
            p[i].fy = 0.0;
            p[i].fz = 0.0;
            
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double dx = p[j].x - p[i].x;
                    double dy = p[j].y - p[i].y;
                    double dz = p[j].z - p[i].z;
                    double r2 = dx*dx + dy*dy + dz*dz;
                    double r = sqrt(r2);
                    
                    if (r > 0.01) {
                        double force = p[i].mass * p[j].mass / (r2 + 0.01);
                        p[i].fx += force * dx / r;
                        p[i].fy += force * dy / r;
                        p[i].fz += force * dz / r;
                    }
                }
            }
        }
    }
}

void ParticleSimulation::integrate() {
    Particle* p = particles;
    double timestep = dt;
    
    #pragma omp target map(tofrom: p[0:N])
    {
        for (int i = 0; i < N; i++) {
            p[i].vx += p[i].fx / p[i].mass * timestep;
            p[i].vy += p[i].fy / p[i].mass * timestep;
            p[i].vz += p[i].fz / p[i].mass * timestep;
            
            p[i].x += p[i].vx * timestep;
            p[i].y += p[i].vy * timestep;
            p[i].z += p[i].vz * timestep;
        }
    }
}

void ParticleSimulation::applyBoundaryConditions() {
    Particle* p = particles;
    int n = N;
    double min_bound = 0.0;
    double max_bound = box_size;
    
    #pragma omp target
    {
        for (int i = 0; i < n; i++) {
            if (p[i].x < min_bound) {
                p[i].x = min_bound;
                p[i].vx = -p[i].vx;
            }
            if (p[i].x > max_bound) {
                p[i].x = max_bound;
                p[i].vx = -p[i].vx;
            }
            if (p[i].y < min_bound) {
                p[i].y = min_bound;
                p[i].vy = -p[i].vy;
            }
            if (p[i].y > max_bound) {
                p[i].y = max_bound;
                p[i].vy = -p[i].vy;
            }
            if (p[i].z < min_bound) {
                p[i].z = min_bound;
                p[i].vz = -p[i].vz;
            }
            if (p[i].z > max_bound) {
                p[i].z = max_bound;
                p[i].vz = -p[i].vz;
            }
        }
    }
}

double ParticleSimulation::calculateEnergy() {
    std::vector<double> energies(N);
    double total_energy = 0.0;
    
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < N; i++) {
        double ke = 0.5 * particles[i].mass * 
                   (particles[i].vx * particles[i].vx + 
                    particles[i].vy * particles[i].vy + 
                    particles[i].vz * particles[i].vz);
        energies[i] = ke;
    }
    
    for (int i = 0; i < N; i++) {
        total_energy += energies[i];
    }
    
    return total_energy;
}

void ParticleSimulation::updateNeighborList() {
    Particle* p = particles;
    int* nlist = neighbor_list;
    int* ncount = neighbor_count;
    int n = N;
    double cutoff = 2.0;
    
    #pragma omp target teams distribute parallel for map(to: p[0:n]) map(from: nlist[0:n*50], ncount[0:n])
    for (int i = 0; i < n; i++) {
        ncount[i] = 0;
        int prev_neighbor = -1;
        
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dx = p[j].x - p[i].x;
                double dy = p[j].y - p[i].y;
                double dz = p[j].z - p[i].z;
                double r2 = dx*dx + dy*dy + dz*dz;
                
                if (r2 < cutoff * cutoff) {
                    if (prev_neighbor >= 0 && j < prev_neighbor) {
                        continue;
                    }
                    nlist[i * 50 + ncount[i]] = j;
                    ncount[i]++;
                    prev_neighbor = j;
                }
            }
        }
    }
}

void ParticleSimulation::runSimulation(int steps) {
    std::cout << "Starting simulation with " << N << " particles for " << steps << " steps" << std::endl;
    
    for (int step = 0; step < steps; step++) {
        #pragma omp target update to(particles[0:N]) nowait
        
        calculateForces();
        
        #pragma omp target update from(particles[0:N]) nowait
        
        integrate();
        
        applyBoundaryConditions();
        
        if (step % 10 == 0) {
            updateNeighborList();
        }
        
        if (step % 100 == 0) {
            double energy = calculateEnergy();
            std::cout << "Step " << step << " Energy: " << energy << std::endl;
        }
    }
}

int main() {
    const int N = 2048;
    const int num_steps = 1000;
    
    std::cout << "Initializing particle simulation..." << std::endl;
    
    ParticleSimulation sim(N);
    
    std::cout << "Running simulation..." << std::endl;
    sim.runSimulation(num_steps);
    
    std::cout << "Simulation complete!" << std::endl;
    
    return 0;
}