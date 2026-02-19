/*
 * Sensor Logger Application
 * Purpose: Simulates reading and logging sensor data on embedded devices
 * Target: ARM-based embedded Linux systems
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_READINGS 5
#define TEMP_BASE 20.0
#define HUMIDITY_BASE 45.0

/*
 * Simulate reading temperature sensor
 * Returns: Temperature value in Celsius
 */
float read_temperature() {
    return TEMP_BASE + ((float)(rand() % 100) / 10.0);
}

/*
 * Simulate reading humidity sensor
 * Returns: Humidity percentage
 */
float read_humidity() {
    return HUMIDITY_BASE + ((float)(rand() % 200) / 10.0);
}

int main() {
    time_t current_time;
    struct tm *time_info;
    char time_buffer[80];
    float temperature, humidity;
    int i;
    
    printf("Sensor Logger v1.0 starting...\n");
    printf("Initializing sensor interfaces...\n");
    
    // Seed random number generator for simulated sensor data
    srand(time(NULL));
    
    // Main sensor reading loop
    for (i = 0; i < NUM_READINGS; i++) {
        // Get current timestamp
        time(&current_time);
        time_info = localtime(&current_time);
        strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", time_info);
        
        // Read sensor values
        temperature = read_temperature();
        humidity = read_humidity();
        
        // Log the readings
        printf("[%s] Reading #%d: Temperature=%.2f°C, Humidity=%.2f%%\n",
               time_buffer, i + 1, temperature, humidity);
        
        // Small delay simulation
        sleep(1);
    }
    
    printf("Sensor Logger shutting down...\n");
    printf("Total readings logged: %d\n", NUM_READINGS);
    
    return 0;
}