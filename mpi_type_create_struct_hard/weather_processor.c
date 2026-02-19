// Parallel Weather Data Processor
// Distributes weather station readings across MPI processes for analysis

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define NUM_STATIONS 10
#define CSV_FILE "data/stations.csv"

// Weather record structure containing sensor measurements
typedef struct {
    int station_id;
    double temperature;
    float humidity;
    double pressure;
    float wind_speed;
} WeatherRecord;

int main(int argc, char** argv) {
    int rank, size;
    WeatherRecord* records;
    
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Allocate memory for weather records
    records = (WeatherRecord*)malloc(NUM_STATIONS * sizeof(WeatherRecord));
    if (records == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Process 0 reads the weather data from CSV file
    if (rank == 0) {
        FILE* fp = fopen(CSV_FILE, "r");
        if (fp == NULL) {
            fprintf(stderr, "Error: Could not open %s\n", CSV_FILE);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Skip the CSV header line
        char header[256];
        if (fgets(header, sizeof(header), fp) == NULL) {
            fprintf(stderr, "Error reading CSV header\n");
            fclose(fp);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Read weather station data
        for (int i = 0; i < NUM_STATIONS; i++) {
            if (fscanf(fp, "%d,%lf,%f,%lf,%f\n",
                      &records[i].station_id,
                      &records[i].temperature,
                      &records[i].humidity,
                      &records[i].pressure,
                      &records[i].wind_speed) != 5) {
                fprintf(stderr, "Error parsing CSV line %d\n", i + 1);
                fclose(fp);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        
        fclose(fp);
        printf("Process 0: Successfully loaded %d weather station records\n", NUM_STATIONS);
    }
    
    // Broadcast weather data to all processes
    // BUG: Using MPI_INT instead of proper handling of the struct
    // This causes data corruption because MPI_INT interprets the bytes incorrectly
    MPI_Bcast(records, NUM_STATIONS * sizeof(WeatherRecord) / sizeof(int), 
              MPI_INT, 0, MPI_COMM_WORLD);
    
    // All processes verify and print received data
    printf("\n=== Process %d: Received Weather Data ===\n", rank);
    for (int i = 0; i < NUM_STATIONS; i++) {
        printf("Station %d: Temp=%.2f°C, Humidity=%.1f%%, Pressure=%.2f hPa, Wind=%.1f m/s\n",
               records[i].station_id,
               records[i].temperature,
               records[i].humidity,
               records[i].pressure,
               records[i].wind_speed);
    }
    
    // Clean up
    free(records);
    MPI_Finalize();
    
    return 0;
}