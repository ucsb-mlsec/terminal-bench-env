#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "config_types.h"

/* Device Manager - Legacy embedded systems configuration
 * WARNING: This file uses pre-C99 initialization patterns
 * that need to be refactored for modern compilers
 */

// Initialize the device registry with default values
DeviceRegistry init_device_registry(void) {
    DeviceRegistry registry;
    // Old style positional initialization
    DeviceRegistry temp = {10, 0, NULL, {"registry_v1", 1, 0}};
    registry = temp;
    return registry;
}

// Create a default device entry
DeviceInfo create_device_entry(void) {
    DeviceInfo device;
    // Pre-C99 positional member initialization
    DeviceInfo temp = {"DEV-UNKNOWN", "v0.0.1", 0, 0};
    device = temp;
    return device;
}

// Setup GPIO configuration with default values
GPIOConfig setup_gpio_config(void) {
    GPIOConfig config;
    // Old style initialization without designated initializers
    GPIOConfig temp = {0, 0, 0, 0, 3300};
    config = temp;
    return config;
}

// Initialize power configuration
PowerConfig init_power_config(void) {
    PowerConfig power;
    // Legacy positional initialization pattern
    PowerConfig temp = {5000, 3300, 500, 10000, 1};
    power = temp;
    return power;
}

// Register multiple devices using old initialization style
void register_devices(DeviceInfo devices[]) {
    // Old C89 style array initialization without designated initializers
    DeviceInfo temp_devices[3] = {
        {"DEV-SENSOR-01", "v1.2.3", 1, 100},
        {"DEV-ACTUATOR-02", "v2.0.1", 1, 200},
        {"DEV-CONTROLLER-03", "v1.5.0", 0, 150}
    };
    
    memcpy(devices, temp_devices, sizeof(DeviceInfo) * 3);
}

// Update device operational status
int update_device_status(DeviceInfo *device, int status) {
    if (device == NULL) {
        return -1;
    }
    
    device->status = status;
    
    if (status == 1) {
        printf("Device %s is now ACTIVE\n", device->device_id);
    } else {
        printf("Device %s is now INACTIVE\n", device->device_id);
    }
    
    return 0;
}

// Print device information to stdout
void print_device_info(const DeviceInfo *device) {
    if (device == NULL) {
        printf("Error: NULL device pointer\n");
        return;
    }
    
    printf("=== Device Information ===\n");
    printf("Device ID: %s\n", device->device_id);
    printf("Firmware Version: %s\n", device->firmware_version);
    printf("Status: %s\n", device->status ? "ACTIVE" : "INACTIVE");
    printf("Power Draw: %u mW\n", device->power_draw_mw);
    printf("========================\n");
}

// Check device health status
int check_device_health(const DeviceInfo *device, const PowerConfig *power_config) {
    if (device == NULL || power_config == NULL) {
        return -1;
    }
    
    // Check if device is active
    if (device->status == 0) {
        printf("WARNING: Device %s is inactive\n", device->device_id);
        return 0;
    }
    
    // Check power consumption
    if (device->power_draw_mw > power_config->max_power_mw) {
        printf("ERROR: Device %s exceeds maximum power draw\n", device->device_id);
        return -2;
    }
    
    if (device->power_draw_mw < power_config->idle_power_mw) {
        printf("WARNING: Device %s power draw below idle threshold\n", device->device_id);
        return 0;
    }
    
    // Device health is good
    printf("Device %s health: OK\n", device->device_id);
    return 1;
}

// Initialize system with default configurations
int init_system_configs(SystemConfig *sys_config) {
    if (sys_config == NULL) {
        return -1;
    }
    
    // Old style nested struct initialization
    SystemConfig temp = {
        {10, 0, NULL, {"registry_v1", 1, 0}},
        {5000, 3300, 500, 10000, 1},
        {0, 0, 0, 0, 3300}
    };
    
    *sys_config = temp;
    
    printf("System configuration initialized\n");
    printf("Max devices: %u\n", sys_config->registry.max_devices);
    printf("System voltage: %u mV\n", sys_config->power.system_voltage_mv);
    printf("GPIO reference voltage: %u mV\n", sys_config->gpio.vref_mv);
    
    return 0;
}

// Validate power configuration parameters
int validate_power_config(const PowerConfig *config) {
    if (config == NULL) {
        return 0;
    }
    
    if (config->system_voltage_mv < config->min_voltage_mv) {
        printf("ERROR: System voltage below minimum\n");
        return 0;
    }
    
    if (config->max_power_mw < config->idle_power_mw) {
        printf("ERROR: Max power less than idle power\n");
        return 0;
    }
    
    return 1;
}

// Get device count from registry
uint32_t get_device_count(const DeviceRegistry *registry) {
    if (registry == NULL) {
        return 0;
    }
    return registry->device_count;
}