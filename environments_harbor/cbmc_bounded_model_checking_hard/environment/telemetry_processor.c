#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#define MAX_SAMPLES 10
#define FILTER_SIZE 5
#define OUTPUT_BUFFER 8
#define VALIDATION_SIZE 7
#define CALIBRATION_POINTS 12

int main() {
    // Declare buffers for telemetry processing
    uint16_t raw_data[MAX_SAMPLES];
    uint16_t filtered_data[OUTPUT_BUFFER];
    uint16_t validation_buffer[VALIDATION_SIZE];
    uint16_t calibration_table[CALIBRATION_POINTS];
    
    // Stage 1: Data Acquisition - Read sensor telemetry data
    // Simulates reading from sensors into buffer
    for (int i = 0; i < MAX_SAMPLES; i++) {
        assert(i >= 0);
        assert(i < MAX_SAMPLES);
        raw_data[i] = (uint16_t)(i * 100 + 50);
    }
    
    // Stage 2: Calibration Table Initialization
    // Longest single loop - requires unwinding of 12
    for (int c = 0; c < CALIBRATION_POINTS; c++) {
        assert(c >= 0);
        assert(c < CALIBRATION_POINTS);
        calibration_table[c] = (uint16_t)(c * 10);
    }
    
    // Stage 3: Data Filtering with Moving Average
    // Nested loops: outer loop iterates OUTPUT_BUFFER times
    // Inner loop iterates FILTER_SIZE times
    for (int i = 0; i < OUTPUT_BUFFER; i++) {
        assert(i >= 0);
        assert(i < OUTPUT_BUFFER);
        
        uint32_t sum = 0;
        for (int j = 0; j < FILTER_SIZE; j++) {
            assert(j >= 0);
            assert(j < FILTER_SIZE);
            assert(i + j < MAX_SAMPLES);
            
            // Apply filter kernel
            sum += raw_data[i + j];
        }
        
        // Store filtered result
        filtered_data[i] = (uint16_t)(sum / FILTER_SIZE);
    }
    
    // Stage 4: Data Validation and Checksum
    // Validate filtered data meets quality constraints
    for (int v = 0; v < VALIDATION_SIZE; v++) {
        assert(v >= 0);
        assert(v < VALIDATION_SIZE);
        assert(v < OUTPUT_BUFFER);
        
        validation_buffer[v] = filtered_data[v];
        
        // Check data is within valid telemetry range
        assert(validation_buffer[v] < 65535);
    }
    
    // Stage 5: Calibration Application
    // Apply calibration to validated data
    for (int k = 0; k < VALIDATION_SIZE; k++) {
        assert(k >= 0);
        assert(k < VALIDATION_SIZE);
        
        // Simple calibration offset
        uint16_t calibrated = validation_buffer[k] + calibration_table[k];
        assert(calibrated >= validation_buffer[k] || calibration_table[k] == 0);
    }
    
    return 0;
}