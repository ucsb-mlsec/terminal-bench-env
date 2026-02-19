#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 4096
#define MAX_ITER 10000
#define TOLERANCE 1e-5
#define ALPHA 0.1

// Host-only function for debugging
void print_debug_info(const char* msg) {
    printf("DEBUG: %s\n", msg);
}

// Function with missing map clauses entirely - data_mapping issue #1
void stencil_computation(float* T_old, float* T_new, int n) {
    #pragma omp target teams distribute parallel for
    for (int i = 1; i < n - 1; i++) {
        T_new[i] = T_old[i] + ALPHA * (T_old[i-1] - 2.0f * T_old[i] + T_old[i+1]);
    }
}

// Function with incorrect array section syntax - data_mapping issue #2
void apply_boundary_conditions(float* T, int n, float left_temp, float right_temp) {
    #pragma omp target map(T[0])
    {
        T[0] = left_temp;
        T[n-1] = right_temp;
    }
}

// Function with reduction but not using proper target reduction - parallelization issue #1
float check_convergence(float* T_old, float* T_new, int n) {
    float diff = 0.0f;
    #pragma omp target map(to: T_old[0:n], T_new[0:n])
    {
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < n; i++) {
            float local_diff = fabs(T_new[i] - T_old[i]);
            diff += local_diff;
        }
    }
    return diff / n;
}

// Function with improper thread hierarchy - parallelization issue #2
void compute_laplacian(float* T, float* lap, int n) {
    #pragma omp target parallel map(to: T[0:n]) map(from: lap[0:n])
    {
        #pragma omp for
        for (int i = 1; i < n - 1; i++) {
            lap[i] = T[i-1] - 2.0f * T[i] + T[i+1];
        }
    }
}

// Function calling host-only function inside target - target_structure issue #1
void initialize_temperature(float* T, int n) {
    #pragma omp target map(from: T[0:n])
    {
        for (int i = 0; i < n; i++) {
            T[i] = 0.0f;
            if (i == n/2) {
                print_debug_info("Initializing center point");
            }
        }
    }
}

// Function not handling device data consistency - dependencies issue #1
void swap_buffers(float** T_old, float** T_new) {
    float* temp = *T_old;
    *T_old = *T_new;
    *T_new = temp;
}

// Additional helper function with data mapping
void set_initial_conditions(float* T, int n) {
    for (int i = 0; i < n; i++) {
        if (i > n/4 && i < 3*n/4) {
            T[i] = 100.0f;
        } else {
            T[i] = 0.0f;
        }
    }
}

// Function to compute energy
float compute_total_energy(float* T, int n) {
    float energy = 0.0f;
    #pragma omp target teams distribute parallel for map(to: T[0:n]) reduction(+:energy)
    for (int i = 0; i < n; i++) {
        energy += T[i] * T[i];
    }
    return energy;
}

// Main simulation function
void run_simulation(float* T_old, float* T_new, int n, int max_iter) {
    float left_bc = 0.0f;
    float right_bc = 0.0f;
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Apply boundary conditions
        apply_boundary_conditions(T_old, n, left_bc, right_bc);
        
        // Perform stencil computation
        stencil_computation(T_old, T_new, n);
        
        // Check convergence every 100 iterations
        if (iter % 100 == 0) {
            float conv = check_convergence(T_old, T_new, n);
            if (conv < TOLERANCE) {
                printf("Converged at iteration %d\n", iter);
                break;
            }
        }
        
        // Swap buffers
        swap_buffers(&T_old, &T_new);
    }
}

int main(int argc, char** argv) {
    int n = N;
    int max_iterations = MAX_ITER;
    
    printf("Starting 1D Heat Transfer Simulation\n");
    printf("Grid size: %d\n", n);
    printf("Max iterations: %d\n", max_iterations);
    
    // Allocate memory
    float* T_old = (float*)malloc(n * sizeof(float));
    float* T_new = (float*)malloc(n * sizeof(float));
    float* laplacian = (float*)malloc(n * sizeof(float));
    
    if (!T_old || !T_new || !laplacian) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize temperature distribution
    printf("Initializing temperature field...\n");
    set_initial_conditions(T_old, n);
    memcpy(T_new, T_old, n * sizeof(float));
    
    // Try to initialize on device (has issues)
    initialize_temperature(T_old, n);
    
    // Compute initial energy
    float initial_energy = compute_total_energy(T_old, n);
    printf("Initial energy: %f\n", initial_energy);
    
    // Run main simulation
    printf("Running simulation...\n");
    double start_time = omp_get_wtime();
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Apply boundary conditions
        apply_boundary_conditions(T_old, n, 0.0f, 0.0f);
        
        // Compute stencil
        stencil_computation(T_old, T_new, n);
        
        // Compute laplacian periodically
        if (iter % 50 == 0) {
            compute_laplacian(T_old, laplacian, n);
        }
        
        // Check convergence
        if (iter % 100 == 0) {
            float convergence = check_convergence(T_old, T_new, n);
            printf("Iteration %d: convergence = %e\n", iter, convergence);
            
            if (convergence < TOLERANCE) {
                printf("Simulation converged at iteration %d\n", iter);
                break;
            }
        }
        
        // Swap buffers
        swap_buffers(&T_old, &T_new);
    }
    
    double end_time = omp_get_wtime();
    printf("Simulation completed in %f seconds\n", end_time - start_time);
    
    // Compute final energy
    float final_energy = compute_total_energy(T_old, n);
    printf("Final energy: %f\n", final_energy);
    
    // Cleanup
    free(T_old);
    free(T_new);
    free(laplacian);
    
    return 0;
}