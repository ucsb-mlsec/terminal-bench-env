#ifndef SENSOR_DATA_H
#define SENSOR_DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <float.h>

#define READINGS_PER_SENSOR 100
#define MAX_SENSORS 4
#define FILENAME "input/sensors.txt"
#define OUTPUT_FILE "results.txt"

typedef struct {
    double temperatures[READINGS_PER_SENSOR];
    int count;
    double sum;
    double avg;
    double min;
    double max;
    int sensor_id;
} SensorData;

#endif