#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_File fh;
    int data[256];
    int i;
    MPI_Offset offset;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) {
            fprintf(stderr, "This program requires exactly 4 processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Initialize data array with values 0-255
    for (i = 0; i < 256; i++) {
        data[i] = i;
    }

    // Calculate offset for this process
    offset = rank * 1024;

    // Open file for writing
    MPI_File_open(MPI_COMM_WORLD, "/tmp/output.dat", 
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, 
                  MPI_INFO_NULL, &fh);

    // Write data at the specified offset
    MPI_File_write_at(fh, offset, data, 256, MPI_INT, &status);

    // Close the file
    MPI_File_close(&fh);

    MPI_Finalize();
    return 0;
}