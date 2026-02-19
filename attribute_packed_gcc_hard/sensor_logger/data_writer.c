#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "sensor_data.h"

int write_sensor_data(const char* input_file, const char* output_file) {
    FILE* input_fp = fopen(input_file, "r");
    if (!input_fp) {
        fprintf(stderr, "Error: Cannot open input file %s\n", input_file);
        return -1;
    }

    FILE* output_fp = fopen(output_file, "wb");
    if (!output_fp) {
        fprintf(stderr, "Error: Cannot open output file %s\n", output_file);
        fclose(input_fp);
        return -1;
    }

    int record_count = 0;
    char line[256];

    while (fgets(line, sizeof(line), input_fp)) {
        sensor_record record;
        
        // Parse comma-separated values
        char* token = strtok(line, ",");
        if (!token) continue;
        record.sensor_id = (uint8_t)atoi(token);

        token = strtok(NULL, ",");
        if (!token) continue;
        record.timestamp = (uint64_t)strtoull(token, NULL, 10);

        token = strtok(NULL, ",");
        if (!token) continue;
        record.temperature = (float)atof(token);

        token = strtok(NULL, ",");
        if (!token) continue;
        record.humidity = (float)atof(token);

        token = strtok(NULL, ",");
        if (!token) continue;
        record.pressure = (uint16_t)atoi(token);

        // Write the struct to binary file
        size_t written = fwrite(&record, sizeof(sensor_record), 1, output_fp);
        if (written != 1) {
            fprintf(stderr, "Error: Failed to write record %d\n", record_count);
            break;
        }

        record_count++;
    }

    fclose(input_fp);
    fclose(output_fp);

    return record_count;
}

int main() {
    int records_written = write_sensor_data("test_data.txt", "output.bin");
    
    if (records_written < 0) {
        fprintf(stderr, "Error: Failed to write sensor data\n");
        return 1;
    }

    printf("Records written: %d\n", records_written);
    printf("Size of sensor_record: %zu bytes\n", sizeof(sensor_record));

    return 0;
}