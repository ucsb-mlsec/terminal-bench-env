/*
 * Sensor Filter Module - Safety-Critical Embedded System
 * 
 * This module implements a multi-stage filtering algorithm for sensor data.
 * SAFETY-CRITICAL: Used in control systems where incorrect filtering can
 * lead to hazardous conditions.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

/* Configuration Constants */
#define MAX_SENSORS 10
#define BUFFER_SIZE 100
#define FILTER_WINDOW 5
#define MIN_VALID_VALUE -100.0f
#define MAX_VALID_VALUE 100.0f
#define OUTLIER_THRESHOLD 50.0f

/* Verification macro for formal analysis */
#define VERIFY(cond) assert(cond)

/* Sensor Data Structure */
typedef struct {
    int sensor_id;
    float value;
    uint32_t timestamp;
    bool valid;
} SensorData;

/* Global state for filtering */
static float sensor_history[MAX_SENSORS][FILTER_WINDOW];
static int history_index[MAX_SENSORS];
static int history_count[MAX_SENSORS];

/*
 * SAFETY-CRITICAL FUNCTION: Initialize sensor history buffers
 * Must be called before processing any sensor data
 */
void init_sensor_filter(void) {
    for (int i = 0; i < MAX_SENSORS; i++) {
        history_index[i] = 0;
        history_count[i] = 0;
        for (int j = 0; j < FILTER_WINDOW; j++) {
            sensor_history[i][j] = 0.0f;
        }
    }
}

/*
 * SAFETY-CRITICAL FUNCTION: Validate input sensor data
 * Returns: true if data passes all safety checks
 */
bool validate_sensor_data(const SensorData *data) {
    VERIFY(data != NULL);
    
    /* Check sensor ID is within valid range */
    if (data->sensor_id < 0 || data->sensor_id >= MAX_SENSORS) {
        return false;
    }
    
    /* Check value is within safe operating bounds */
    if (data->value < MIN_VALID_VALUE || data->value > MAX_VALID_VALUE) {
        return false;
    }
    
    /* Verify timestamp is reasonable (non-zero) */
    if (data->timestamp == 0) {
        return false;
    }
    
    return true;
}

/*
 * SAFETY-CRITICAL FUNCTION: Calculate moving average
 * Computes average over the filter window to smooth sensor noise
 */
float calculate_moving_average(int sensor_id) {
    VERIFY(sensor_id >= 0 && sensor_id < MAX_SENSORS);
    
    float sum = 0.0f;
    int count = history_count[sensor_id];
    
    /* Ensure we have data to average */
    VERIFY(count > 0);
    
    /* Sum all values in the history window */
    for (int i = 0; i < count; i++) {
        VERIFY(i < FILTER_WINDOW); /* Bound check for verification */
        sum += sensor_history[sensor_id][i];
    }
    
    /* SAFETY CHECK: Prevent division by zero */
    VERIFY(count != 0);
    
    return sum / (float)count;
}

/*
 * SAFETY-CRITICAL FUNCTION: Detect outliers in sensor readings
 * Returns: true if value is an outlier that should be rejected
 */
bool is_outlier(float current_value, float average) {
    float deviation = current_value - average;
    if (deviation < 0) {
        deviation = -deviation;  /* Absolute value */
    }
    
    return deviation > OUTLIER_THRESHOLD;
}

/*
 * SAFETY-CRITICAL FUNCTION: Main filtering algorithm
 * Processes sensor data through validation, averaging, and outlier detection
 * Returns: filtered value or -1.0f on error
 */
float process_sensor_data(const SensorData *data, float *filtered_output) {
    VERIFY(data != NULL);
    VERIFY(filtered_output != NULL);
    
    /* Stage 1: Input Validation */
    if (!validate_sensor_data(data)) {
        return -1.0f;  /* Invalid input */
    }
    
    int id = data->sensor_id;
    VERIFY(id >= 0 && id < MAX_SENSORS);
    
    /* Stage 2: Update history buffer with new reading */
    int idx = history_index[id];
    
    /* POTENTIAL SAFETY VIOLATION: Off-by-one error in bound check */
    /* This should be caught by verification! */
    VERIFY(idx >= 0 && idx <= FILTER_WINDOW);  /* BUG: Should be < not <= */
    
    sensor_history[id][idx] = data->value;
    history_index[id] = (idx + 1) % FILTER_WINDOW;
    
    if (history_count[id] < FILTER_WINDOW) {
        history_count[id]++;
    }
    
    /* Stage 3: Calculate moving average (requires sufficient history) */
    if (history_count[id] < 2) {
        *filtered_output = data->value;
        return data->value;
    }
    
    float average = calculate_moving_average(id);
    
    /* Stage 4: Outlier detection */
    if (is_outlier(data->value, average)) {
        /* Use previous average instead of outlier value */
        *filtered_output = average;
    } else {
        *filtered_output = data->value;
    }
    
    /* Stage 5: Final safety check on output */
    VERIFY(*filtered_output >= MIN_VALID_VALUE);
    VERIFY(*filtered_output <= MAX_VALID_VALUE);
    
    return *filtered_output;
}