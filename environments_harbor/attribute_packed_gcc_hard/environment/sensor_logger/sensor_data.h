#ifndef SENSOR_DATA_H
#define SENSOR_DATA_H

#include <stdint.h>

struct sensor_record {
    uint32_t sensor_id;
    uint64_t timestamp;
    float temperature;
    float humidity;
    float pressure;
};

int write_sensor_data(const char* input_file, const char* output_file);

#endif