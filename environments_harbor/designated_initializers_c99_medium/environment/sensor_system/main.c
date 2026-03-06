#include <stdio.h>
#include <stdlib.h>
#include "sensor.h"

int main() {
    printf("Sensor System v1.0 - Processing 5 readings\n");
    
    struct SensorConfig config = {5, 25.0, 60.0, "Building A"};
    
    struct SensorReading readings[5] = {
        {1, 22.5, 55.0, 1609459200, 'A'},
        {2, 23.1, 58.5, 1609459260, 'A'},
        {3, 24.8, 62.0, 1609459320, 'W'},
        {4, 21.9, 54.5, 1609459380, 'A'},
        {5, 25.5, 65.0, 1609459440, 'E'}
    };
    
    process_readings(readings, 5, &config);
    
    return 0;
}