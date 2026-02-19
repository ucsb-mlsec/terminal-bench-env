/*
 * Automotive Safety-Critical System - Brake Control Module
 * Verification Target for CPAchecker Analysis
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MAX_SENSORS 4
#define MAX_PRESSURE 1000
#define MIN_PRESSURE 0
#define BUFFER_SIZE 10

/* Global state variables */
int sensor_readings[MAX_SENSORS];
int pressure_history[BUFFER_SIZE];
int history_index = 0;

/*
 * Read sensor data from brake pressure sensors
 * Safety-critical: Must validate sensor index
 */
int sensor_read(int sensor_id) {
    // Safety check: sensor index must be within valid range
    assert(sensor_id >= 0 && sensor_id < MAX_SENSORS);
    
    // Simulated sensor reading
    int reading = sensor_readings[sensor_id];
    
    // Safety check: reading must be within physical limits
    assert(reading >= MIN_PRESSURE && reading <= MAX_PRESSURE);
    
    return reading;
}

/*
 * Validate input pressure value
 * Returns 1 if valid, 0 if invalid
 */
int validate_input(int pressure) {
    if (pressure < MIN_PRESSURE) {
        return 0;
    }
    if (pressure > MAX_PRESSURE) {
        return 0;
    }
    return 1;
}

/*
 * Store pressure reading in circular buffer
 * Safety-critical: Buffer overflow protection
 */
void store_pressure_history(int pressure) {
    // Safety check: history index must be valid
    assert(history_index >= 0 && history_index < BUFFER_SIZE);
    
    pressure_history[history_index] = pressure;
    history_index = (history_index + 1) % BUFFER_SIZE;
    
    // Post-condition: index remains within bounds
    assert(history_index >= 0 && history_index < BUFFER_SIZE);
}

/*
 * Process sensor data and compute average pressure
 * Safety-critical: Array access bounds checking
 */
int process_data(int *data, int count) {
    // Safety check: pointer must not be null
    assert(data != NULL);
    
    // Safety check: count must be valid
    assert(count > 0 && count <= MAX_SENSORS);
    
    int sum = 0;
    int i;
    
    for (i = 0; i < count; i++) {
        // Safety check: array access within bounds
        assert(i >= 0 && i < MAX_SENSORS);
        
        int value = data[i];
        
        // Safety check: each value must be validated
        assert(validate_input(value) == 1);
        
        sum += value;
    }
    
    int average = sum / count;
    
    // Safety check: average must be within valid range
    assert(average >= MIN_PRESSURE && average <= MAX_PRESSURE);
    
    return average;
}

/*
 * Control brake actuator based on processed pressure
 * Safety-critical: Emergency brake activation logic
 */
void control_actuator(int pressure, int emergency_mode) {
    // Safety check: pressure must be validated
    assert(validate_input(pressure) == 1);
    
    if (emergency_mode) {
        // Emergency mode: apply maximum braking
        assert(pressure >= 0);
        printf("EMERGENCY: Maximum brake applied\n");
    } else {
        // Normal mode: proportional braking
        if (pressure > 800) {
            printf("High pressure braking\n");
        } else if (pressure > 400) {
            printf("Medium pressure braking\n");
        } else {
            printf("Low pressure braking\n");
        }
    }
    
    // Store in history for analysis
    store_pressure_history(pressure);
}

/*
 * Main control loop for brake system
 */
int main(void) {
    int i;
    
    // Initialize sensor readings with safe values
    for (i = 0; i < MAX_SENSORS; i++) {
        assert(i >= 0 && i < MAX_SENSORS);
        sensor_readings[i] = 500; // Mid-range pressure
    }
    
    // Initialize pressure history buffer
    for (i = 0; i < BUFFER_SIZE; i++) {
        assert(i >= 0 && i < BUFFER_SIZE);
        pressure_history[i] = 0;
    }
    
    // Safety-critical section: Read all sensors
    printf("Reading brake pressure sensors...\n");
    for (i = 0; i < MAX_SENSORS; i++) {
        int reading = sensor_read(i);
        assert(reading >= MIN_PRESSURE && reading <= MAX_PRESSURE);
    }
    
    // Process sensor data
    printf("Processing sensor data...\n");
    int avg_pressure = process_data(sensor_readings, MAX_SENSORS);
    
    // Validate processed result
    assert(avg_pressure >= MIN_PRESSURE && avg_pressure <= MAX_PRESSURE);
    
    // Control actuator based on processed data
    int emergency = (avg_pressure > 900) ? 1 : 0;
    control_actuator(avg_pressure, emergency);
    
    printf("Brake control cycle completed successfully\n");
    
    return 0;
}