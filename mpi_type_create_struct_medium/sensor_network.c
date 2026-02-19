#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int sensor_id;
    float voltage;
    int timestamp;
    char status;
} SensorReading;

int main(int argc, char** argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This program requires exactly 2 processes\n");
        }
        MPI_Finalize();
        exit(1);
    }
    
    SensorReading reading;
    
    if (rank == 0) {
        // Process 0: Sender
        reading.sensor_id = 101;
        reading.voltage = 3.7;
        reading.timestamp = 1234567890;
        reading.status = 'A';
        
        // THE BUG: Using MPI_BYTE with sizeof doesn't handle structure padding correctly
        MPI_Send(&reading, sizeof(SensorReading), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        
        printf("Process 0: Sent sensor data - ID=%d, Voltage=%.1f, Timestamp=%d, Status=%c\n",
               reading.sensor_id, reading.voltage, reading.timestamp, reading.status);
    } 
    else if (rank == 1) {
        // Process 1: Receiver
        SensorReading received;
        memset(&received, 0, sizeof(SensorReading));
        
        // THE BUG: Also using MPI_BYTE with sizeof - matches sender's mistake
        MPI_Recv(&received, sizeof(SensorReading), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        printf("Process 1: Received sensor data - ID=%d, Voltage=%.1f, Timestamp=%d, Status=%c\n",
               received.sensor_id, received.voltage, received.timestamp, received.status);
        
        // Verify the data
        int id_match = (received.sensor_id == 101);
        int voltage_match = (received.voltage > 3.69 && received.voltage < 3.71);
        
        FILE* json_file = fopen("/workspace/result.json", "w");
        if (json_file != NULL) {
            fprintf(json_file, "{\n");
            fprintf(json_file, "  \"status\": \"%s\",\n", 
                    (id_match && voltage_match) ? "success" : "failed");
            fprintf(json_file, "  \"sensor_id\": %d,\n", received.sensor_id);
            fprintf(json_file, "  \"voltage\": %.1f\n", received.voltage);
            fprintf(json_file, "}\n");
            fclose(json_file);
        }
    }
    
    MPI_Finalize();
    return 0;
}