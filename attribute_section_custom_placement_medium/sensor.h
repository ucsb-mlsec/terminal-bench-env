#ifndef SENSOR_H
#define SENSOR_H

#include <stdint.h>

typedef struct {
    float offset;
    float scale;
    int precision;
    char description[64];
} CalibrationData;

typedef struct {
    unsigned int sensor_id;
    unsigned int hardware_id;
    char name[32];
} SensorMapping;

typedef void (*SensorCallback)(int sensor_id, float value);

typedef struct {
    SensorCallback handlers[8];
    int count;
} CallbackRegistry;

void init_sensor_lib(void);
int process_sensor_data(int id, float value);

#endif