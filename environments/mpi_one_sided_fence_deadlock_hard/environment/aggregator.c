#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "sensor_data.h"

#define READINGS_PER_SENSOR 100
#define NUM_SENSORS 4

typedef struct {
    double sum;
    double avg;
    double min;
    double max;
    int count;
} SensorStats;

int main(int argc, char** argv) {
    int rank, size;
    MPI_Win win;
    SensorStats *local_stats, *global_stats = NULL;
    double *readings;
    FILE *fp;
    char filename[] = "input/sensors.txt";
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NUM_SENSORS) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires exactly 4 MPI processes\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Allocate memory for readings and local statistics
    readings = (double*)malloc(READINGS_PER_SENSOR * sizeof(double));
    local_stats = (SensorStats*)malloc(sizeof(SensorStats));

    // Read sensor data from file
    fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Rank %d: Error opening file\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Skip to the appropriate section for this rank
    for (i = 0; i < rank * READINGS_PER_SENSOR; i++) {
        double dummy;
        fscanf(fp, "%lf", &dummy);
    }

    // Read this process's data
    for (i = 0; i < READINGS_PER_SENSOR; i++) {
        fscanf(fp, "%lf", &readings[i]);
    }
    fclose(fp);

    // Calculate local statistics
    local_stats->sum = 0.0;
    local_stats->min = readings[0];
    local_stats->max = readings[0];
    local_stats->count = READINGS_PER_SENSOR;

    for (i = 0; i < READINGS_PER_SENSOR; i++) {
        local_stats->sum += readings[i];
        if (readings[i] < local_stats->min) local_stats->min = readings[i];
        if (readings[i] > local_stats->max) local_stats->max = readings[i];
    }
    local_stats->avg = local_stats->sum / local_stats->count;

    printf("Rank %d: Local avg=%.2f, min=%.2f, max=%.2f\n", 
           rank, local_stats->avg, local_stats->min, local_stats->max);

    // Create window for one-sided communication
    if (rank == 0) {
        global_stats = (SensorStats*)malloc(NUM_SENSORS * sizeof(SensorStats));
    }
    
    MPI_Win_create(global_stats, 
                   rank == 0 ? NUM_SENSORS * sizeof(SensorStats) : 0,
                   sizeof(SensorStats), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // One-sided communication: all processes put their stats to rank 0
    MPI_Win_fence(0, win);
    
    MPI_Put(local_stats, sizeof(SensorStats), MPI_BYTE,
            0, rank * sizeof(SensorStats), sizeof(SensorStats), MPI_BYTE, win);

    // BUG: Missing MPI_Win_fence here after MPI_Put operations
    // This causes the program to hang because the next communication phase
    // starts before the previous one is properly synchronized

    // Rank 0 performs additional operations
    if (rank == 0) {
        // Try to access the data - but without proper synchronization
        double total_sum = 0.0;
        double global_min = global_stats[0].min;
        double global_max = global_stats[0].max;
        
        for (i = 0; i < NUM_SENSORS; i++) {
            total_sum += global_stats[i].sum;
            if (global_stats[i].min < global_min) global_min = global_stats[i].min;
            if (global_stats[i].max > global_max) global_max = global_stats[i].max;
        }

        FILE *out = fopen("results.txt", "w");
        fprintf(out, "Temperature Monitoring System - Aggregated Results\n");
        fprintf(out, "==================================================\n");
        fprintf(out, "Total sensors: %d\n", NUM_SENSORS);
        fprintf(out, "Readings per sensor: %d\n", READINGS_PER_SENSOR);
        fprintf(out, "Global average: %.2f\n", total_sum / (NUM_SENSORS * READINGS_PER_SENSOR));
        fprintf(out, "Global minimum: %.2f\n", global_min);
        fprintf(out, "Global maximum: %.2f\n", global_max);
        fclose(out);
        
        printf("Results written to results.txt\n");
    }

    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    free(readings);
    free(local_stats);
    if (rank == 0) {
        free(global_stats);
    }

    MPI_Finalize();
    return 0;
}