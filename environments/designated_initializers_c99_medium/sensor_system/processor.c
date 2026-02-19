#include "sensor.h"
#include <stdio.h>
#include <string.h>

void process_readings(struct SensorReading *readings, int count, struct SensorConfig config)
{
    struct SensorStats stats = {0.0, 0.0, 0, 0};
    
    for (int i = 0; i < count; i++) {
        stats.avg_temp += readings[i].temperature;
        stats.avg_humidity += readings[i].humidity;
        stats.total_readings++;
        if (readings[i].status == 'E' || readings[i].status == 'W') {
            stats.error_count++;
        }
    }
    
    stats.avg_temp /= count;
    stats.avg_humidity /= count;
    
    printf("Processing %d readings from %s\n", count, config.location);
    print_stats(stats);
}

void print_stats(struct SensorStats stats)
{
    printf("Average Temperature: %.1f°C\n", stats.avg_temp);
    printf("Average Humidity: %.1f%%\n", stats.avg_humidity);
    printf("Total Readings: %d\n", stats.total_readings);
    printf("Errors/Warnings: %d\n", stats.error_count);
}