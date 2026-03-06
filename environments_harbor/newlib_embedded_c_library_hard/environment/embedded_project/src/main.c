#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Data logging application for ARM Cortex-M4 */
/* This application runs continuously in a bare-metal embedded environment */

#define MAX_LOG_ENTRIES 100
#define SENSOR_COUNT 4

/* Log entry structure */
typedef struct {
    uint32_t timestamp;
    uint16_t sensor_id;
    int16_t temperature;
    uint16_t humidity;
    uint16_t pressure;
    uint8_t status;
} LogEntry;

/* Sensor reading structure */
typedef struct {
    uint16_t sensor_id;
    int16_t raw_value;
    uint8_t valid;
} SensorReading;

/* Global pointers for dynamically allocated buffers */
static LogEntry *log_buffer = NULL;
static uint8_t *temp_buffer = NULL;
static SensorReading *sensor_data = NULL;
static char *message_buffer = NULL;

/* External syscall functions defined in syscalls.c */
extern int _write(int file, char *ptr, int len);
extern void *_sbrk(int incr);

/* Forward declarations */
void init_system(void);
void log_data(uint32_t timestamp);
SensorReading process_sensor_reading(uint16_t sensor_id);
int store_to_buffer(LogEntry *entry);

/* System initialization function */
void init_system(void) {
    /* Allocate main log buffer - 1024 bytes */
    log_buffer = (LogEntry *)malloc(MAX_LOG_ENTRIES * sizeof(LogEntry));
    if (log_buffer == NULL) {
        /* Failed to allocate memory */
        while(1);
    }
    memset(log_buffer, 0, MAX_LOG_ENTRIES * sizeof(LogEntry));
    
    /* Allocate temporary processing buffer - 2048 bytes */
    temp_buffer = (uint8_t *)malloc(2048);
    if (temp_buffer == NULL) {
        while(1);
    }
    
    /* Allocate sensor data buffer - 512 bytes */
    sensor_data = (SensorReading *)malloc(SENSOR_COUNT * sizeof(SensorReading) * 16);
    if (sensor_data == NULL) {
        while(1);
    }
    
    /* Allocate message formatting buffer - 512 bytes */
    message_buffer = (char *)malloc(512);
    if (message_buffer == NULL) {
        while(1);
    }
    
    /* Initialize sensor data structures */
    for (int i = 0; i < SENSOR_COUNT; i++) {
        sensor_data[i].sensor_id = i;
        sensor_data[i].raw_value = 0;
        sensor_data[i].valid = 0;
    }
}

/* Process a sensor reading and return structured data */
SensorReading process_sensor_reading(uint16_t sensor_id) {
    SensorReading reading;
    reading.sensor_id = sensor_id;
    
    /* Simulate sensor reading - in real hardware this would read from ADC/I2C/SPI */
    reading.raw_value = (int16_t)((sensor_id * 123 + 456) % 1000);
    reading.valid = 1;
    
    /* Use temp buffer for processing */
    if (temp_buffer != NULL) {
        memcpy(temp_buffer, &reading, sizeof(SensorReading));
        /* Additional processing could be done here */
    }
    
    return reading;
}

/* Store log entry to buffer */
int store_to_buffer(LogEntry *entry) {
    static uint32_t buffer_index = 0;
    
    if (log_buffer == NULL || entry == NULL) {
        return -1;
    }
    
    /* Circular buffer behavior */
    memcpy(&log_buffer[buffer_index], entry, sizeof(LogEntry));
    buffer_index = (buffer_index + 1) % MAX_LOG_ENTRIES;
    
    return 0;
}

/* Main logging function */
void log_data(uint32_t timestamp) {
    LogEntry entry;
    
    /* Allocate temporary entry structure on heap for processing */
    LogEntry *temp_entry = (LogEntry *)malloc(sizeof(LogEntry));
    if (temp_entry == NULL) {
        return;
    }
    
    /* Process each sensor */
    for (uint16_t i = 0; i < SENSOR_COUNT; i++) {
        SensorReading reading = process_sensor_reading(i);
        
        /* Build log entry */
        temp_entry->timestamp = timestamp;
        temp_entry->sensor_id = reading.sensor_id;
        temp_entry->temperature = reading.raw_value;
        temp_entry->humidity = reading.raw_value + 100;
        temp_entry->pressure = reading.raw_value + 200;
        temp_entry->status = reading.valid;
        
        /* Store to main buffer */
        store_to_buffer(temp_entry);
        
        /* Format message for output */
        if (message_buffer != NULL) {
            snprintf(message_buffer, 512, 
                     "Sensor %d: Temp=%d, Humidity=%d, Pressure=%d\n",
                     temp_entry->sensor_id,
                     temp_entry->temperature,
                     temp_entry->humidity,
                     temp_entry->pressure);
        }
    }
    
    free(temp_entry);
}

/* Simulate hardware delay */
void delay(uint32_t count) {
    for (volatile uint32_t i = 0; i < count; i++) {
        __asm__("nop");
    }
}

/* Main application entry point */
int main(void) {
    uint32_t timestamp = 0;
    uint32_t iteration = 0;
    
    /* Initialize system and allocate buffers */
    init_system();
    
    /* Infinite logging loop - typical for embedded systems */
    while (1) {
        /* Perform data logging */
        log_data(timestamp);
        
        /* Periodically allocate additional buffers to test heap behavior */
        if (iteration % 10 == 0) {
            /* Allocate temporary analysis buffer */
            uint8_t *analysis_buf = (uint8_t *)malloc(256);
            if (analysis_buf != NULL) {
                /* Use buffer for analysis */
                memset(analysis_buf, 0xAA, 256);
                /* Free buffer */
                free(analysis_buf);
            }
        }
        
        /* Simulate time passing */
        timestamp++;
        iteration++;
        
        /* Delay between logging cycles */
        delay(100000);
        
        /* In field deployment, this runs for extended periods */
        /* Memory corruption occurs after many iterations */
    }
    
    /* Should never reach here in embedded system */
    return 0;
}