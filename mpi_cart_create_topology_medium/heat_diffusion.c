#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    int ndims = 2;
    int dims[2];
    int periods[2];
    int reorder = 1;
    MPI_Comm cart_comm;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check that we have exactly 16 processes
    if (size != 16) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires exactly 16 MPI processes\n");
            fprintf(stderr, "Please run with: mpirun -np 16 ./heat_diffusion\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Set up 4x4 grid topology
    // Bug: dimensions are set incorrectly - should be [4, 4] but order matters
    dims[0] = 4;
    dims[1] = 4;
    
    // Enable periodic boundaries
    // Bug: periods array not properly initialized for wraparound
    periods[0] = 0;  // Should be 1 for periodic boundary
    periods[1] = 0;  // Should be 1 for periodic boundary
    
    // Create Cartesian topology
    // Bug: parameters might be in wrong order or incorrectly specified
    int result = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 
                                  reorder, &cart_comm);
    
    if (result != MPI_SUCCESS) {
        if (rank == 0) {
            fprintf(stderr, "Error: Failed to create Cartesian topology\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Get the coordinates of this process in the grid
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, ndims, coords);
    
    // Verify the topology was created successfully
    int topo_type;
    MPI_Topo_test(cart_comm, &topo_type);
    
    if (topo_type != MPI_CART) {
        if (rank == 0) {
            fprintf(stderr, "Error: Topology is not Cartesian\n");
        }
        MPI_Comm_free(&cart_comm);
        MPI_Finalize();
        return 1;
    }
    
    // Only rank 0 writes the status file
    if (rank == 0) {
        FILE* fp = fopen("/workspace/topology_status.txt", "w");
        if (fp == NULL) {
            fprintf(stderr, "Error: Could not open status file for writing\n");
            MPI_Comm_free(&cart_comm);
            MPI_Finalize();
            return 1;
        }
        
        fprintf(fp, "processes=%d\n", size);
        fprintf(fp, "grid=%dx%d\n", dims[0], dims[1]);
        fprintf(fp, "success=true\n");
        
        fclose(fp);
        
        printf("Heat diffusion simulation initialized successfully\n");
        printf("Grid topology: %dx%d with periodic boundaries\n", dims[0], dims[1]);
        printf("Total processes: %d\n", size);
    }
    
    // Synchronize all processes
    MPI_Barrier(cart_comm);
    
    // Clean up
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    
    return 0;
}