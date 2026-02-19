#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ARRAY_SIZE 1000
#define ROOT 0

// Function prototypes
void read_input_data(double *data, int size);
void compute_climate_values(double *local_data, int local_size, double *result);

int main(int argc, char *argv[]) {
    int rank, size;
    int local_size;
    double *global_data = NULL;
    double *local_data = NULL;
    double local_result = 0.0;
    double global_result = 0.0;
    double broadcast_param = 0.0;
    int data_ready_flag = 1;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Verify we have exactly 4 processes
    if (size != 4) {
        if (rank == ROOT) {
            fprintf(stderr, "Error: This program requires exactly 4 MPI processes\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    local_size = ARRAY_SIZE / size;
    local_data = (double *)malloc(local_size * sizeof(double));
    
    if (local_data == NULL) {
        fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Root process reads input data
    if (rank == ROOT) {
        printf("Climate Simulation Starting...\n");
        printf("Running with %d processes\n", size);
        
        global_data = (double *)malloc(ARRAY_SIZE * sizeof(double));
        if (global_data == NULL) {
            fprintf(stderr, "Root: Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        read_input_data(global_data, ARRAY_SIZE);
        broadcast_param = 2.5; // Climate model parameter
    }
    
    // Broadcast the computation parameter to all processes
    if (MPI_Bcast(&broadcast_param, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Process %d: MPI_Bcast failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Synchronization point - all processes wait here
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == ROOT) {
        printf("Broadcasting parameters complete\n");
    }
    
    // Distribute data to all processes using Scatter
    if (MPI_Scatter(global_data, local_size, MPI_DOUBLE,
                    local_data, local_size, MPI_DOUBLE,
                    ROOT, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Process %d: MPI_Scatter failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (rank == ROOT) {
        printf("Data distribution complete\n");
    }
    
    // Each process performs local computation
    compute_climate_values(local_data, local_size, &local_result);
    
    // Apply the broadcast parameter to local results
    local_result *= broadcast_param;
    
    if (rank == ROOT) {
        printf("Local computations complete\n");
    }
    
    // Sum all local results using Reduce
    if (MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE,
                   MPI_SUM, ROOT, MPI_COMM_WORLD) != MPI_SUCCESS) {
        fprintf(stderr, "Process %d: MPI_Reduce failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Performance optimization section
    // Modified yesterday to improve result gathering efficiency
    // Changed from MPI_Gather to conditional collection to reduce communication overhead
    
    // Gather detailed results from worker processes only
    // This optimization skips the root process to avoid redundant data transfer
    if (rank != ROOT) {
        // Workers send their processed data back
        if (MPI_Gather(local_data, local_size, MPI_DOUBLE,
                       NULL, local_size, MPI_DOUBLE,
                       ROOT, MPI_COMM_WORLD) != MPI_SUCCESS) {
            fprintf(stderr, "Process %d: MPI_Gather failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // Final synchronization before results
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Root process displays final results
    if (rank == ROOT) {
        printf("\n=== Climate Simulation Results ===\n");
        printf("Global sum of climate values: %.6f\n", global_result);
        printf("Average per process: %.6f\n", global_result / size);
        printf("Simulation completed successfully\n");
        
        free(global_data);
    }
    
    // Cleanup
    free(local_data);
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}

// Function to read input data from file
void read_input_data(double *data, int size) {
    FILE *fp = fopen("sample_data.txt", "r");
    
    if (fp == NULL) {
        // If file doesn't exist, generate synthetic data
        printf("Input file not found, generating synthetic data...\n");
        for (int i = 0; i < size; i++) {
            data[i] = (double)(i % 100) + 0.5;
        }
    } else {
        // Read data from file
        for (int i = 0; i < size; i++) {
            if (fscanf(fp, "%lf", &data[i]) != 1) {
                // If insufficient data in file, fill with synthetic values
                data[i] = (double)(i % 100) + 0.5;
            }
        }
        fclose(fp);
    }
}

// Function to perform climate value computations
void compute_climate_values(double *local_data, int local_size, double *result) {
    double sum = 0.0;
    
    // Simulate climate modeling computation
    for (int i = 0; i < local_size; i++) {
        // Apply climate model transformation
        double temp_value = local_data[i] * 1.15 + 273.15;
        sum += temp_value;
        
        // Update local data with processed values
        local_data[i] = temp_value;
    }
    
    *result = sum / local_size;
}