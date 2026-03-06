#ifndef SENSOR_H
#define SENSOR_H

struct SensorReading {
    int id;
    float temperature;
    float humidity;
    long timestamp;
    char status;
};

struct SensorConfig {
    int sensor_count;
    float temp_threshold;
    float humidity_threshold;
    char *location;
};

struct SensorStats {
    float avg_temp;
    float avg_humidity;
    int total_readings;
    int error_count;
};

void process_readings(struct SensorReading *readings, int count, struct SensorConfig config);
void print_stats(struct SensorStats stats);

#endif