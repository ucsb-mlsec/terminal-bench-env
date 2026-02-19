#include <omp.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define NATOMS 8192

struct Atom {
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;
};

class MolecularDynamics {
private:
    Atom* atoms;
    double box_size;
    double cutoff;
    double dt;
    double temperature;
    int* neighbor_list;
    int* neighbor_count;
    
public:
    MolecularDynamics(double box, double cut, double timestep) 
        : box_size(box), cutoff(cut), dt(timestep), temperature(300.0) {
        atoms = new Atom[NATOMS];
        neighbor_list = new int[NATOMS * 100];
        neighbor_count = new int[NATOMS];
        initialize_positions();
        initialize_velocities();
    }
    
    ~MolecularDynamics() {
        delete[] atoms;
        delete[] neighbor_list;
        delete[] neighbor_count;
    }
    
    void initialize_positions() {
        int n = (int)ceil(pow(NATOMS, 1.0/3.0));
        double spacing = box_size / n;
        int idx = 0;
        for(int i = 0; i < n && idx < NATOMS; i++) {
            for(int j = 0; j < n && idx < NATOMS; j++) {
                for(int k = 0; k < n && idx < NATOMS; k++) {
                    atoms[idx].x = i * spacing;
                    atoms[idx].y = j * spacing;
                    atoms[idx].z = k * spacing;
                    idx++;
                }
            }
        }
    }
    
    void initialize_velocities() {
        for(int i = 0; i < NATOMS; i++) {
            atoms[i].vx = ((double)rand()/RAND_MAX - 0.5) * 2.0;
            atoms[i].vy = ((double)rand()/RAND_MAX - 0.5) * 2.0;
            atoms[i].vz = ((double)rand()/RAND_MAX - 0.5) * 2.0;
        }
    }
    
    void compute_forces() {
        double cutoff_sq = cutoff * cutoff;
        double sigma = 1.0;
        double epsilon = 1.0;
        double sigma6 = pow(sigma, 6.0);
        double sigma12 = sigma6 * sigma6;
        
        for(int i = 0; i < NATOMS; i++) {
            atoms[i].fx = 0.0;
            atoms[i].fy = 0.0;
            atoms[i].fz = 0.0;
        }
        
        // Issue 1: Missing map clause for force arrays (data_mapping issue)
        #pragma omp target
        {
            for(int i = 0; i < NATOMS-1; i++) {
                for(int j = i+1; j < NATOMS; j++) {
                    double dx = atoms[i].x - atoms[j].x;
                    double dy = atoms[i].y - atoms[j].y;
                    double dz = atoms[i].z - atoms[j].z;
                    
                    double r2 = dx*dx + dy*dy + dz*dz;
                    
                    if(r2 < cutoff_sq) {
                        double r2inv = 1.0 / r2;
                        double r6inv = r2inv * r2inv * r2inv;
                        double force_mag = 24.0 * epsilon * r6inv * (2.0 * sigma12 * r6inv - sigma6) * r2inv;
                        
                        atoms[i].fx += force_mag * dx;
                        atoms[i].fy += force_mag * dy;
                        atoms[i].fz += force_mag * dz;
                        atoms[j].fx -= force_mag * dx;
                        atoms[j].fy -= force_mag * dy;
                        atoms[j].fz -= force_mag * dz;
                    }
                }
            }
        }
    }
    
    void velocity_verlet_step1() {
        // Issue 2: target teams without parallel for directive (parallelization issue)
        #pragma omp target teams map(tofrom: atoms[0:NATOMS])
        {
            for(int i = 0; i < NATOMS; i++) {
                atoms[i].vx += 0.5 * atoms[i].fx * dt;
                atoms[i].vy += 0.5 * atoms[i].fy * dt;
                atoms[i].vz += 0.5 * atoms[i].fz * dt;
                
                atoms[i].x += atoms[i].vx * dt;
                atoms[i].y += atoms[i].vy * dt;
                atoms[i].z += atoms[i].vz * dt;
                
                atoms[i].x = fmod(atoms[i].x + box_size, box_size);
                atoms[i].y = fmod(atoms[i].y + box_size, box_size);
                atoms[i].z = fmod(atoms[i].z + box_size, box_size);
            }
        }
    }
    
    void velocity_verlet_step2() {
        #pragma omp target teams distribute parallel for map(tofrom: atoms[0:NATOMS])
        for(int i = 0; i < NATOMS; i++) {
            atoms[i].vx += 0.5 * atoms[i].fx * dt;
            atoms[i].vy += 0.5 * atoms[i].fy * dt;
            atoms[i].vz += 0.5 * atoms[i].fz * dt;
        }
    }
    
    double compute_kinetic_energy() {
        double ke = 0.0;
        
        // Issue 3: Missing proper map for scalar result (data_mapping issue)
        #pragma omp target teams distribute parallel for reduction(+:ke)
        for(int i = 0; i < NATOMS; i++) {
            ke += 0.5 * (atoms[i].vx * atoms[i].vx + 
                        atoms[i].vy * atoms[i].vy + 
                        atoms[i].vz * atoms[i].vz);
        }
        
        return ke;
    }
    
    void build_neighbor_list() {
        double cutoff_neighbor = cutoff * 1.2;
        double cutoff_neighbor_sq = cutoff_neighbor * cutoff_neighbor;
        
        for(int i = 0; i < NATOMS; i++) {
            neighbor_count[i] = 0;
        }
        
        // Issue 4: Nested loops without proper collapse/parallelization (parallelization issue)
        #pragma omp target map(to: atoms[0:NATOMS]) map(tofrom: neighbor_list[0:NATOMS*100], neighbor_count[0:NATOMS])
        {
            for(int i = 0; i < NATOMS; i++) {
                int count = 0;
                for(int j = 0; j < NATOMS; j++) {
                    if(i != j) {
                        double dx = atoms[i].x - atoms[j].x;
                        double dy = atoms[i].y - atoms[j].y;
                        double dz = atoms[i].z - atoms[j].z;
                        double r2 = dx*dx + dy*dy + dz*dz;
                        
                        if(r2 < cutoff_neighbor_sq) {
                            neighbor_list[i * 100 + count] = j;
                            count++;
                        }
                    }
                }
                neighbor_count[i] = count;
            }
        }
    }
    
    void scale_temperature(double target_temp) {
        double current_ke = compute_kinetic_energy();
        double current_temp = (2.0 * current_ke) / (3.0 * NATOMS);
        double scale_factor = sqrt(target_temp / current_temp);
        
        // Issue 5: Accessing class member variable temperature not properly mapped (target_structure issue)
        #pragma omp target teams distribute parallel for
        for(int i = 0; i < NATOMS; i++) {
            atoms[i].vx *= scale_factor;
            atoms[i].vy *= scale_factor;
            atoms[i].vz *= scale_factor;
            
            double vel_mag = sqrt(atoms[i].vx * atoms[i].vx + 
                                 atoms[i].vy * atoms[i].vy + 
                                 atoms[i].vz * atoms[i].vz);
            
            if(vel_mag > temperature * 0.1) {
                atoms[i].vx *= 0.9;
                atoms[i].vy *= 0.9;
                atoms[i].vz *= 0.9;
            }
        }
    }
    
    double* compute_pair_distances() {
        double* distances = new double[NATOMS];
        
        // Issue 6: Loop-carried dependency not handled (dependencies issue)
        #pragma omp target teams distribute parallel for map(to: atoms[0:NATOMS]) map(from: distances[0:NATOMS])
        for(int i = 0; i < NATOMS; i++) {
            if(i > 0) {
                double dx = atoms[i].x - atoms[i-1].x;
                double dy = atoms[i].y - atoms[i-1].y;
                double dz = atoms[i].z - atoms[i-1].z;
                distances[i] = sqrt(dx*dx + dy*dy + dz*dz);
            } else {
                distances[i] = 0.0;
            }
        }
        
        return distances;
    }
    
    // Issue 7: Incorrect defaultmap usage (target_structure issue)
    void apply_boundary_conditions() {
        #pragma omp target defaultmap(tofrom:scalar) map(tofrom: atoms[0:NATOMS])
        #pragma omp teams distribute parallel for
        for(int i = 0; i < NATOMS; i++) {
            if(atoms[i].x < 0.0) atoms[i].x += box_size;
            if(atoms[i].x >= box_size) atoms[i].x -= box_size;
            if(atoms[i].y < 0.0) atoms[i].y += box_size;
            if(atoms[i].y >= box_size) atoms[i].y -= box_size;
            if(atoms[i].z < 0.0) atoms[i].z += box_size;
            if(atoms[i].z >= box_size) atoms[i].z -= box_size;
        }
    }
    
    void run_simulation(int nsteps) {
        for(int step = 0; step < nsteps; step++) {
            compute_forces();
            velocity_verlet_step1();
            compute_forces();
            velocity_verlet_step2();
            
            if(step % 100 == 0) {
                double ke = compute_kinetic_energy();
                double temp = (2.0 * ke) / (3.0 * NATOMS);
                std::cout << "Step " << step << " KE: " << ke << " T: " << temp << std::endl;
            }
            
            if(step % 50 == 0) {
                build_neighbor_list();
            }
            
            if(step % 200 == 0) {
                scale_temperature(300.0);
            }
        }
    }
};

int main() {
    std::cout << "Starting Molecular Dynamics Simulation" << std::endl;
    std::cout << "Number of atoms: " << NATOMS << std::endl;
    
    double box_size = 50.0;
    double cutoff = 2.5;
    double dt = 0.001;
    
    MolecularDynamics md(box_size, cutoff, dt);
    
    md.run_simulation(1000);
    
    std::cout << "Simulation complete" << std::endl;
    
    return 0;
}