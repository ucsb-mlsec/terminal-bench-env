#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int *data = NULL;
    int data_count = 0;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check command-line arguments
    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    char *input_filename = argv[1];
    char *output_filename = argv[2];
    
    // Process 0 reads the input file
    if (rank == 0) {
        FILE *input_file = fopen(input_filename, "r");
        if (input_file == NULL) {
            fprintf(stderr, "Error: Cannot open input file %s\n", input_filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // First pass: count the number of integers
        int temp;
        data_count = 0;
        while (fscanf(input_file, "%d", &temp) == 1) {
            data_count++;
        }
        
        // Allocate memory for data
        data = (int *)malloc(data_count * sizeof(int));
        if (data == NULL) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            fclose(input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Second pass: read the integers
        rewind(input_file);
        for (int i = 0; i < data_count; i++) {
            if (fscanf(input_file, "%d", &data[i]) != 1) {
                fprintf(stderr, "Error: Failed to read data\n");
                free(data);
                fclose(input_file);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        
        fclose(input_file);
        
        printf("Read %d values from input file\n", data_count);
    }
    
    // Broadcast the data count to all processes
    MPI_Bcast(&data_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // ====================================================================
    // TODO: Add parallel prefix sum computation here
    // 
    // Steps to implement:
    // 1. Distribute the data array across all processes
    //    - Calculate how many elements each process should handle
    //    - Handle cases where data_count is not evenly divisible by size
    //    - Use MPI_Scatter or MPI_Scatterv to distribute data
    //
    // 2. Each process computes local prefix sum on its portion
    //    - Compute cumulative sum within the local array
    //
    // 3. Adjust for global offsets
    //    - Each process needs to add the sum of all elements from 
    //      previous processes to its local results
    //    - Use MPI_Scan or implement custom communication pattern
    //
    // 4. Gather results back to process 0
    //    - Use MPI_Gather or MPI_Gatherv to collect results
    //
    // Example allocation for local data on each process:
    // int local_count = ...; // calculate based on rank and data_count
    // int *local_data = (int *)malloc(local_count * sizeof(int));
    // int *local_result = (int *)malloc(local_count * sizeof(int));
    // 
    // ====================================================================
    
    // Process 0 writes the output file
    if (rank == 0) {
        FILE *output_file = fopen(output_filename, "w");
        if (output_file == NULL) {
            fprintf(stderr, "Error: Cannot open output file %s\n", output_filename);
            free(data);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // TODO: Replace this with actual prefix sum results
        // This is just a placeholder showing how to write output
        // In the actual implementation, you should write the 
        // computed prefix sum values here
        for (int i = 0; i < data_count; i++) {
            fprintf(output_file, "%d\n", data[i]); // PLACEHOLDER - replace with prefix sum
        }
        
        fclose(output_file);
        printf("Wrote %d values to output file\n", data_count);
        
        // Free the data array
        free(data);
    }
    
    // Clean up MPI
    MPI_Finalize();
    
    return 0;
}