#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define GRID_SIZE 10000
#define NUM_ITERATIONS 100

/* 
 * Climate Simulation Application - Hybrid MPI+OpenMP Implementation
 * 
 * This application simulates large-scale climate data processing using
 * a hybrid parallel approach:
 * - MPI for distributed memory parallelism across nodes
 * - OpenMP for shared memory parallelism within each node
 * 
 * Each MPI process handles a subdomain of the global climate grid.
 * Within each process, multiple OpenMP threads perform computations
 * and communicate with neighboring processes independently.
 */

/* Function to process climate data with hybrid parallelization */
void process_climate_data(double *temperature, double *humidity, 
                          int local_size, int rank, int size) {
    int i, iter;
    double *recv_buffer = (double *)malloc(local_size * sizeof(double));
    
    printf("Rank %d: Starting climate data processing with %d threads\n", 
           rank, omp_get_max_threads());
    
    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        
        /* 
         * OpenMP parallel region where multiple threads perform computation
         * and MPI communication concurrently.
         * 
         * CRITICAL: Multiple threads make MPI calls simultaneously without
         * any serialization mechanism. This requires MPI_THREAD_MULTIPLE
         * support from the MPI implementation.
         */
        #pragma omp parallel private(i)
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            int chunk_size = local_size / num_threads;
            int start = thread_id * chunk_size;
            int end = (thread_id == num_threads - 1) ? local_size : start + chunk_size;
            
            /* Each thread processes its own chunk of data */
            for (i = start; i < end; i++) {
                temperature[i] = temperature[i] * 0.99 + humidity[i] * 0.01;
                humidity[i] = humidity[i] * 0.98 + temperature[i] * 0.02;
            }
            
            /*
             * Each thread independently communicates with neighboring processes
             * Multiple threads make MPI calls concurrently without locks
             * This is the key pattern requiring MPI_THREAD_MULTIPLE
             */
            if (rank > 0) {
                /* Send data to previous rank - each thread sends its chunk */
                MPI_Send(&temperature[start], chunk_size, MPI_DOUBLE, 
                        rank - 1, thread_id, MPI_COMM_WORLD);
                
                /* Receive boundary data from previous rank */
                MPI_Recv(recv_buffer, chunk_size, MPI_DOUBLE, 
                        rank - 1, thread_id + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                /* Update local data with received boundary information */
                for (i = start; i < end; i++) {
                    temperature[i] = (temperature[i] + recv_buffer[i - start]) * 0.5;
                }
            }
            
            if (rank < size - 1) {
                /* Receive data from next rank */
                MPI_Recv(recv_buffer, chunk_size, MPI_DOUBLE, 
                        rank + 1, thread_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                /* Send data to next rank - each thread sends independently */
                MPI_Send(&humidity[start], chunk_size, MPI_DOUBLE, 
                        rank + 1, thread_id + 100, MPI_COMM_WORLD);
                
                /* Blend received data */
                for (i = start; i < end; i++) {
                    humidity[i] = (humidity[i] + recv_buffer[i - start]) * 0.5;
                }
            }
        }
        
        /* Synchronization between iterations */
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    free(recv_buffer);
}

/* Function to perform reduction operations with thread-level parallelism */
void compute_global_statistics(double *temperature, int local_size, int rank) {
    double local_sum = 0.0;
    double global_sum = 0.0;
    
    /*
     * Parallel reduction where each thread computes partial sum
     * and then performs MPI communication
     */
    #pragma omp parallel reduction(+:local_sum)
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = local_size / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? local_size : start + chunk_size;
        
        /* Compute local partial sum */
        for (int i = start; i < end; i++) {
            local_sum += temperature[i];
        }
        
        /*
         * Each thread performs its own MPI_Allreduce call
         * This demonstrates concurrent MPI calls from multiple threads
         */
        double thread_global_sum = 0.0;
        MPI_Allreduce(&local_sum, &thread_global_sum, 1, MPI_DOUBLE, 
                     MPI_SUM, MPI_COMM_WORLD);
        
        if (thread_id == 0) {
            global_sum = thread_global_sum;
        }
    }
    
    if (rank == 0) {
        printf("Global average temperature: %f\n", global_sum / (local_size * rank));
    }
}

/* Exchange boundary data with neighboring processes using multiple threads */
void exchange_boundary_data(double *data, int local_size, int rank, int size) {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        /*
         * Multiple threads simultaneously call MPI communication functions
         * Each thread handles different boundary regions
         * No synchronization or serialization between threads
         */
        if (thread_id % 2 == 0 && rank > 0) {
            double boundary_value = data[thread_id];
            MPI_Send(&boundary_value, 1, MPI_DOUBLE, rank - 1, 
                    thread_id, MPI_COMM_WORLD);
        }
        
        if (thread_id % 2 == 1 && rank < size - 1) {
            double recv_value;
            MPI_Recv(&recv_value, 1, MPI_DOUBLE, rank + 1, 
                    thread_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            data[local_size - thread_id - 1] = recv_value;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int local_size;
    double *temperature, *humidity;
    
    /*
     * PROBLEM: Using standard MPI_Init() which doesn't specify
     * thread support requirements. This is insufficient for applications
     * where multiple threads make concurrent MPI calls.
     * 
     * Should use MPI_Init_thread() to request MPI_THREAD_MULTIPLE
     */
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* Configure OpenMP to use multiple threads */
    omp_set_num_threads(12);
    
    if (rank == 0) {
        printf("Climate Simulation starting with %d MPI processes\n", size);
        printf("Each process using %d OpenMP threads\n", omp_get_max_threads());
    }
    
    /* Calculate local domain size */
    local_size = GRID_SIZE / size;
    
    /* Allocate local data arrays */
    temperature = (double *)malloc(local_size * sizeof(double));
    humidity = (double *)malloc(local_size * sizeof(double));
    
    if (temperature == NULL || humidity == NULL) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /* Initialize climate data */
    for (int i = 0; i < local_size; i++) {
        temperature[i] = 20.0 + rank * 0.1 + i * 0.001;
        humidity[i] = 50.0 + rank * 0.5 + i * 0.002;
    }
    
    if (rank == 0) {
        printf("Starting main simulation loop...\n");
    }
    
    /* Main climate simulation processing */
    process_climate_data(temperature, humidity, local_size, rank, size);
    
    /* Compute global statistics */
    compute_global_statistics(temperature, local_size, rank);
    
    /* Final boundary exchange */
    exchange_boundary_data(temperature, local_size, rank, size);
    
    if (rank == 0) {
        printf("Simulation completed successfully\n");
    }
    
    /* Cleanup */
    free(temperature);
    free(humidity);
    
    MPI_Finalize();
    
    return 0;
}