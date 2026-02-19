#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int argc, char *argv[]) {
    // PERFORMANCE BOTTLENECK: Limiting threads to 1
    omp_set_num_threads(1);
    
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <grid_size>\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
    if (N < 2) {
        fprintf(stderr, "Grid size must be at least 2\n");
        return 1;
    }
    
    // Allocate memory for three grids
    double **current = (double **)malloc(N * sizeof(double *));
    double **previous = (double **)malloc(N * sizeof(double *));
    double **next = (double **)malloc(N * sizeof(double *));
    
    for (int i = 0; i < N; i++) {
        current[i] = (double *)malloc(N * sizeof(double));
        previous[i] = (double *)malloc(N * sizeof(double));
        next[i] = (double *)malloc(N * sizeof(double));
    }
    
    // Initialize grids to zero
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            current[i][j] = 0.0;
            previous[i][j] = 0.0;
            next[i][j] = 0.0;
        }
    }
    
    // Set wave source at center
    current[N/2][N/2] = 1.0;
    previous[N/2][N/2] = 1.0;
    
    // Wave propagation parameters
    double c = 0.1; // Wave speed factor
    int time_steps = 750;
    
    // Start timing
    double start_time = omp_get_wtime();
    
    // Main time-stepping loop
    for (int t = 0; t < time_steps; t++) {
        // Update interior points using wave equation
        #pragma omp parallel for
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                double laplacian = current[i-1][j] + current[i+1][j] + 
                                  current[i][j-1] + current[i][j+1] - 
                                  4.0 * current[i][j];
                
                next[i][j] = 2.0 * current[i][j] - previous[i][j] + 
                            c * laplacian;
            }
        }
        
        // Boundary conditions - set edges to zero
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            next[i][0] = 0.0;
            next[i][N-1] = 0.0;
            next[0][i] = 0.0;
            next[N-1][i] = 0.0;
        }
        
        // Rotate grids: previous <- current <- next
        double **temp = previous;
        previous = current;
        current = next;
        next = temp;
    }
    
    // End timing
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;
    
    // Calculate checksum for verification
    double checksum = 0.0;
    #pragma omp parallel for reduction(+:checksum)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            checksum += fabs(current[i][j]);
        }
    }
    
    // Print execution time
    printf("Execution time: %.3f seconds\n", execution_time);
    printf("Checksum: %.6f\n", checksum);
    
    // Sample output for verification
    printf("Sample values:\n");
    printf("  Center [%d][%d]: %.6f\n", N/2, N/2, current[N/2][N/2]);
    printf("  Corner [0][0]: %.6f\n", current[0][0]);
    printf("  Point [%d][%d]: %.6f\n", N/4, N/4, current[N/4][N/4]);
    
    // Free memory
    for (int i = 0; i < N; i++) {
        free(current[i]);
        free(previous[i]);
        free(next[i]);
    }
    free(current);
    free(previous);
    free(next);
    
    return 0;
}