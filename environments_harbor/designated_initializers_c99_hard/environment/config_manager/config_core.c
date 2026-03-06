#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config_types.h"

/* Initialize system configuration with default serial port settings */
SystemConfig init_system_config(void) {
    /* Old style positional initialization - needs refactoring */
    SystemConfig config = {115200, 8, 1, 0, 1};
    return config;
}

/* Initialize network configuration with default IP settings */
NetworkConfig init_network_config(void) {
    /* Pre-C99 positional initialization - order dependent */
    NetworkConfig config = {"192.168.1.100", "255.255.255.0", "192.168.1.1", 8080, 1};
    return config;
}

/* Initialize device information structure */
DeviceInfo init_device_info(void) {
    /* Legacy positional initialization pattern */
    DeviceInfo info = {"EMB-SYS-001", "v2.3.5", 12345, 1};
    return info;
}

/* Create sensor configuration with calibration parameters */
SensorConfig create_sensor_config(void) {
    /* Old C89 style initialization - fragile if struct changes */
    SensorConfig sensor = {0, 100, -40, 85, 5000, 1};
    return sensor;
}

/* Initialize advanced system configuration with extended parameters */
SystemConfig init_advanced_system_config(void) {
    /* Another positional initialization requiring refactoring */
    SystemConfig config = {230400, 8, 2, 1, 0};
    return config;
}

/* Initialize backup network configuration */
NetworkConfig init_backup_network_config(void) {
    /* Legacy style - must be converted to designated initializers */
    NetworkConfig config = {"10.0.0.50", "255.255.255.0", "10.0.0.1", 9090, 0};
    return config;
}

/* Global configuration entries array - old style initialization */
ConfigEntry config_entries[] = {
    /* First entry - primary system configuration */
    {1, "primary_system", 0, {0}},
    /* Second entry - network configuration */
    {2, "network_setup", 1, {0}},
    /* Third entry - sensor array */
    {3, "sensor_config", 2, {0}},
    /* Fourth entry - backup system */
    {4, "backup_system", 0, {0}}
};

/* Additional configuration structures for testing */
static SystemConfig test_configs[] = {
    /* Test configuration 1 - low speed */
    {9600, 8, 1, 0, 1},
    /* Test configuration 2 - high speed */
    {921600, 8, 1, 1, 1},
    /* Test configuration 3 - custom parity */
    {57600, 7, 1, 2, 1}
};

/* Static network configurations for fallback */
static NetworkConfig network_profiles[] = {
    /* Profile 1 - development network */
    {"192.168.0.100", "255.255.255.0", "192.168.0.1", 8000, 1},
    /* Profile 2 - production network */
    {"172.16.1.100", "255.255.0.0", "172.16.1.1", 443, 1}
};

/* Device information database */
static DeviceInfo device_database[] = {
    /* Device 1 */
    {"EMB-DEV-001", "v1.0.0", 10001, 1},
    /* Device 2 */
    {"EMB-DEV-002", "v1.2.3", 10002, 1},
    /* Device 3 */
    {"EMB-DEV-003", "v2.0.0", 10003, 0}
};

/* Sensor configuration presets */
static SensorConfig sensor_presets[] = {
    /* Preset 1 - temperature sensor */
    {0, 255, -50, 100, 1000, 1},
    /* Preset 2 - humidity sensor */
    {1, 100, 0, 100, 2000, 1},
    /* Preset 3 - pressure sensor */
    {2, 1023, 300, 1100, 5000, 1}
};

/* Print system configuration details */
void print_system_config(const SystemConfig *config) {
    printf("System Config:\n");
    printf("  Baud Rate: %d\n", config->baud_rate);
    printf("  Data Bits: %d\n", config->data_bits);
    printf("  Stop Bits: %d\n", config->stop_bits);
    printf("  Parity: %d\n", config->parity);
    printf("  Enabled: %d\n", config->enabled);
}

/* Print network configuration details */
void print_network_config(const NetworkConfig *config) {
    printf("Network Config:\n");
    printf("  IP Address: %s\n", config->ip_address);
    printf("  Subnet Mask: %s\n", config->subnet_mask);
    printf("  Gateway: %s\n", config->gateway);
    printf("  Port: %d\n", config->port);
    printf("  DHCP Enabled: %d\n", config->dhcp_enabled);
}

/* Print device information */
void print_device_info(const DeviceInfo *info) {
    printf("Device Info:\n");
    printf("  Serial Number: %s\n", info->serial_number);
    printf("  Firmware Version: %s\n", info->firmware_version);
    printf("  Device ID: %u\n", info->device_id);
    printf("  Active: %d\n", info->active);
}

/* Print sensor configuration */
void print_sensor_config(const SensorConfig *config) {
    printf("Sensor Config:\n");
    printf("  Channel: %d\n", config->channel);
    printf("  Max Value: %d\n", config->max_value);
    printf("  Min Temp: %d\n", config->min_temp);
    printf("  Max Temp: %d\n", config->max_temp);
    printf("  Sample Rate: %d\n", config->sample_rate_ms);
    printf("  Calibrated: %d\n", config->calibrated);
}

/* Print all configuration entries */
void print_all_configs(void) {
    size_t i;
    
    printf("\n=== Configuration Manager Output ===\n\n");
    
    /* Print main configuration entries */
    printf("Configuration Entries:\n");
    for (i = 0; i < sizeof(config_entries) / sizeof(config_entries[0]); i++) {
        printf("Entry %zu: ID=%d, Name=%s, Type=%d\n", 
               i, config_entries[i].config_id, 
               config_entries[i].config_name, 
               config_entries[i].config_type);
    }
    printf("\n");
    
    /* Print test system configurations */
    printf("Test System Configurations:\n");
    for (i = 0; i < sizeof(test_configs) / sizeof(test_configs[0]); i++) {
        printf("Config %zu: ", i);
        print_system_config(&test_configs[i]);
    }
    printf("\n");
    
    /* Print network profiles */
    printf("Network Profiles:\n");
    for (i = 0; i < sizeof(network_profiles) / sizeof(network_profiles[0]); i++) {
        printf("Profile %zu: ", i);
        print_network_config(&network_profiles[i]);
    }
    printf("\n");
    
    /* Print device database */
    printf("Device Database:\n");
    for (i = 0; i < sizeof(device_database) / sizeof(device_database[0]); i++) {
        printf("Device %zu: ", i);
        print_device_info(&device_database[i]);
    }
    printf("\n");
    
    /* Print sensor presets */
    printf("Sensor Presets:\n");
    for (i = 0; i < sizeof(sensor_presets) / sizeof(sensor_presets[0]); i++) {
        printf("Preset %zu: ", i);
        print_sensor_config(&sensor_presets[i]);
    }
}