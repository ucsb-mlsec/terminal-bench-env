#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Device state structure */
typedef struct {
    uint8_t initialized;
    int16_t sensor_value;
    int16_t actuator_position;
    uint8_t calibration_status;
    uint32_t error_code;
} DeviceState;

/* Global device state */
static DeviceState g_device_state = {0};

/* Operation codes */
#define OP_INITIALIZE    0
#define OP_READ_SENSOR   1
#define OP_WRITE_ACTUATOR 2
#define OP_CALIBRATE     3
#define OP_SHUTDOWN      4
#define OP_GET_STATUS    5
#define OP_RESET_ERRORS  6

/* Return codes */
#define RESULT_SUCCESS          0
#define RESULT_ERROR_NOT_INIT   -1
#define RESULT_ERROR_INVALID_OP -2
#define RESULT_ERROR_INVALID_PARAM -3
#define RESULT_ERROR_STATE      -4

/* Operation handler function type */
typedef int (*operation_handler_t)(int param);

/* Forward declarations of operation handlers */
static int handle_initialize(int param);
static int handle_read_sensor(int param);
static int handle_write_actuator(int param);
static int handle_calibrate(int param);
static int handle_shutdown(int param);
static int handle_get_status(int param);
static int handle_reset_errors(int param);

/* Function pointer dispatch table - this is the problematic pattern for CBMC */
static operation_handler_t operation_table[] = {
    handle_initialize,
    handle_read_sensor,
    handle_write_actuator,
    handle_calibrate,
    handle_shutdown,
    handle_get_status,
    handle_reset_errors
};

#define NUM_OPERATIONS (sizeof(operation_table) / sizeof(operation_table[0]))

/* Initialize operation */
static int handle_initialize(int param) {
    if (g_device_state.initialized) {
        return RESULT_ERROR_STATE;
    }
    
    g_device_state.initialized = 1;
    g_device_state.sensor_value = 0;
    g_device_state.actuator_position = 0;
    g_device_state.calibration_status = 0;
    g_device_state.error_code = 0;
    
    printf("Device initialized\n");
    return RESULT_SUCCESS;
}

/* Read sensor operation */
static int handle_read_sensor(int param) {
    if (!g_device_state.initialized) {
        return RESULT_ERROR_NOT_INIT;
    }
    
    /* Simulate sensor reading */
    g_device_state.sensor_value = (int16_t)(param & 0xFFFF);
    
    printf("Sensor read: %d\n", g_device_state.sensor_value);
    return g_device_state.sensor_value;
}

/* Write actuator operation */
static int handle_write_actuator(int param) {
    if (!g_device_state.initialized) {
        return RESULT_ERROR_NOT_INIT;
    }
    
    if (param < -1000 || param > 1000) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    g_device_state.actuator_position = (int16_t)param;
    
    printf("Actuator set to: %d\n", g_device_state.actuator_position);
    return RESULT_SUCCESS;
}

/* Calibrate operation */
static int handle_calibrate(int param) {
    if (!g_device_state.initialized) {
        return RESULT_ERROR_NOT_INIT;
    }
    
    /* Perform calibration routine */
    g_device_state.calibration_status = 1;
    g_device_state.sensor_value = 0;
    g_device_state.actuator_position = 0;
    
    printf("Device calibrated\n");
    return RESULT_SUCCESS;
}

/* Shutdown operation */
static int handle_shutdown(int param) {
    if (!g_device_state.initialized) {
        return RESULT_ERROR_NOT_INIT;
    }
    
    g_device_state.initialized = 0;
    g_device_state.calibration_status = 0;
    
    printf("Device shutdown\n");
    return RESULT_SUCCESS;
}

/* Get status operation */
static int handle_get_status(int param) {
    if (!g_device_state.initialized) {
        return RESULT_ERROR_NOT_INIT;
    }
    
    printf("Status - Init: %d, Cal: %d, Sensor: %d, Actuator: %d\n",
           g_device_state.initialized,
           g_device_state.calibration_status,
           g_device_state.sensor_value,
           g_device_state.actuator_position);
    
    return RESULT_SUCCESS;
}

/* Reset errors operation */
static int handle_reset_errors(int param) {
    if (!g_device_state.initialized) {
        return RESULT_ERROR_NOT_INIT;
    }
    
    g_device_state.error_code = 0;
    
    printf("Errors reset\n");
    return RESULT_SUCCESS;
}

/* Main dispatch function - uses function pointer table */
int dispatch_operation(int operation_code, int param) {
    /* Bounds check */
    if (operation_code < 0 || operation_code >= NUM_OPERATIONS) {
        return RESULT_ERROR_INVALID_OP;
    }
    
    /* Indirect function call through table - problematic for CBMC */
    operation_handler_t handler = operation_table[operation_code];
    return handler(param);
}

/* Main function demonstrating the dispatch mechanism */
int main(int argc, char *argv[]) {
    int result;
    
    printf("=== Embedded Control System Demo ===\n\n");
    
    /* Initialize device */
    result = dispatch_operation(OP_INITIALIZE, 0);
    if (result != RESULT_SUCCESS) {
        printf("Initialization failed: %d\n", result);
        return 1;
    }
    
    /* Read sensor */
    result = dispatch_operation(OP_READ_SENSOR, 42);
    if (result < 0) {
        printf("Sensor read failed: %d\n", result);
    }
    
    /* Write actuator */
    result = dispatch_operation(OP_WRITE_ACTUATOR, 500);
    if (result != RESULT_SUCCESS) {
        printf("Actuator write failed: %d\n", result);
    }
    
    /* Get status */
    result = dispatch_operation(OP_GET_STATUS, 0);
    
    /* Calibrate */
    result = dispatch_operation(OP_CALIBRATE, 0);
    if (result != RESULT_SUCCESS) {
        printf("Calibration failed: %d\n", result);
    }
    
    /* Reset errors */
    result = dispatch_operation(OP_RESET_ERRORS, 0);
    
    /* Shutdown */
    result = dispatch_operation(OP_SHUTDOWN, 0);
    if (result != RESULT_SUCCESS) {
        printf("Shutdown failed: %d\n", result);
    }
    
    printf("\n=== Demo Complete ===\n");
    return 0;
}