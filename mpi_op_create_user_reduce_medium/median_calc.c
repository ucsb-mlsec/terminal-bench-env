#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    int rank, size;
    int local_data[8];
    int local_sum = 0;
    int global_sum = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Initialize local arrays based on rank
    switch(rank) {
        case 0:
            local_data[0] = 12;
            local_data[1] = 45;
            local_data[2] = 23;
            local_data[3] = 67;
            local_data[4] = 34;
            local_data[5] = 89;
            local_data[6] = 56;
            local_data[7] = 78;
            break;
        case 1:
            local_data[0] = 91;
            local_data[1] = 15;
            local_data[2] = 38;
            local_data[3] = 72;
            local_data[4] = 29;
            local_data[5] = 84;
            local_data[6] = 61;
            local_data[7] = 47;
            break;
        case 2:
            local_data[0] = 53;
            local_data[1] = 26;
            local_data[2] = 94;
            local_data[3] = 41;
            local_data[4] = 68;
            local_data[5] = 19;
            local_data[6] = 75;
            local_data[7] = 32;
            break;
        case 3:
            local_data[0] = 87;
            local_data[1] = 64;
            local_data[2] = 21;
            local_data[3] = 58;
            local_data[4] = 35;
            local_data[5] = 92;
            local_data[6] = 49;
            local_data[7] = 76;
            break;
    }
    
    // Calculate local sum
    for(int i = 0; i < 8; i++) {
        local_sum += local_data[i];
    }
    
    // TODO: This currently only computes sum, not median
    // Need to implement proper median calculation across all processes
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if(rank == 0) {
        // This is incorrect - just printing the average, not median
        double avg = global_sum / 32.0;
        printf("Global average (NOT MEDIAN): %.1f\n", avg);
        // Missing: proper median calculation and file output
    }
    
    MPI_Finalize();
    return 0;
}