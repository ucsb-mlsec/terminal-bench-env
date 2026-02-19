#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Forward declarations for utils.c functions
extern int calculate_checksum(const unsigned char *data, int length);
extern void format_output(char *buffer, int value);
extern int validate_range(int value, int min, int max);
extern void debug_print(const char *message);
extern int unused_crypto_function(int key);

// Forward declarations for data_tables.c functions
extern const int* get_calibration_table(void);
extern int get_table_size(void);
extern const char* get_device_name(void);
extern void unused_table_init(void);

// Active functions used in production
int read_sensor_value(int sensor_id) {
    // Simulate reading from a sensor
    int base_value = 1000 + (sensor_id * 100);
    return base_value + (sensor_id % 50);
}

int process_sensor_data(int raw_value) {
    // Apply calibration
    const int *cal_table = get_calibration_table();
    int table_size = get_table_size();
    
    if (table_size > 0 && cal_table != NULL) {
        int index = raw_value % table_size;
        raw_value += cal_table[index];
    }
    
    // Validate the processed value
    if (!validate_range(raw_value, 0, 10000)) {
        return -1;
    }
    
    return raw_value;
}

void update_system_state(int sensor_value) {
    static int state_counter = 0;
    char output_buffer[128];
    
    state_counter++;
    format_output(output_buffer, sensor_value);
    
    if (state_counter % 10 == 0) {
        debug_print("System state updated");
    }
}

int perform_self_test(void) {
    printf("Starting self-test...\n");
    
    // Test sensor reading
    int sensor_val = read_sensor_value(1);
    if (sensor_val < 0) {
        printf("Sensor read failed\n");
        return -1;
    }
    printf("Sensor test: OK (value=%d)\n", sensor_val);
    
    // Test data processing
    int processed = process_sensor_data(sensor_val);
    if (processed < 0) {
        printf("Data processing failed\n");
        return -1;
    }
    printf("Processing test: OK (value=%d)\n", processed);
    
    // Test calibration table access
    const int *table = get_calibration_table();
    if (table == NULL) {
        printf("Calibration table failed\n");
        return -1;
    }
    printf("Calibration test: OK\n");
    
    // Test checksum calculation
    unsigned char test_data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
    int checksum = calculate_checksum(test_data, 5);
    printf("Checksum test: OK (checksum=%d)\n", checksum);
    
    // Test device identification
    const char *device = get_device_name();
    if (device == NULL) {
        printf("Device name failed\n");
        return -1;
    }
    printf("Device: %s\n", device);
    
    // Test system state update
    update_system_state(processed);
    printf("State update test: OK\n");
    
    printf("All tests passed!\n");
    return 0;
}

void run_normal_operation(void) {
    printf("Running normal operation mode...\n");
    
    for (int i = 0; i < 5; i++) {
        int sensor_val = read_sensor_value(i);
        int processed = process_sensor_data(sensor_val);
        update_system_state(processed);
        printf("Cycle %d: sensor=%d, processed=%d\n", i, sensor_val, processed);
    }
}

// Dead code - never called in production
int legacy_sensor_read(int port, int mode) {
    // Old sensor reading method that was replaced
    int value = 0;
    for (int i = 0; i < port; i++) {
        value += mode * i * 17;
        value = value % 4096;
    }
    return value;
}

// Dead code - planned feature never implemented
void advanced_filtering(int *data, int size, int threshold) {
    // Complex filtering algorithm that was never activated
    for (int i = 0; i < size; i++) {
        if (data[i] > threshold) {
            data[i] = threshold;
        }
        for (int j = 0; j < i; j++) {
            data[i] += data[j] / (i + 1);
        }
    }
    
    // Additional dead computation
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    
    if (sum > 0) {
        for (int i = 0; i < size; i++) {
            data[i] = (data[i] * 1000) / sum;
        }
    }
}

// Dead code - diagnostic function never used
void dump_diagnostic_info(void) {
    printf("=== Diagnostic Information ===\n");
    printf("Memory usage: N/A\n");
    printf("CPU load: N/A\n");
    printf("Uptime: N/A\n");
    
    for (int i = 0; i < 100; i++) {
        printf("Register %d: 0x%08X\n", i, i * 0xDEADBEEF);
    }
    
    printf("=== End Diagnostic ===\n");
}

int main(int argc, char *argv[]) {
    printf("Embedded Application v1.0\n");
    printf("Device: %s\n", get_device_name());
    
    // Check for test mode
    if (argc > 1 && strcmp(argv[1], "--test") == 0) {
        int result = perform_self_test();
        return result == 0 ? 0 : 1;
    }
    
    // Normal operation mode
    run_normal_operation();
    
    return 0;
}