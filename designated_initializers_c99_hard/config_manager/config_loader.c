#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "config_types.h"

/* Configuration loader module for embedded systems
 * Provides default configurations and factory reset functionality
 */

// Load default serial port parameters
SerialParams load_default_serial(void) {
    SerialParams params = {9600, 8, 1, 0, 0, 500};
    return params;
}

// Load default network configuration
NetworkConfig load_default_network(void) {
    NetworkConfig config;
    // Old style initialization - positional
    NetworkConfig temp = {"10.0.0.50", "255.255.0.0", "10.0.0.1", 80, 0};
    config = temp;
    return config;
}

// Load default sensor configuration values
SensorConfig load_sensor_defaults(void) {
    SensorConfig sensor = {1, 255, -273, 1000, 1000, 1};
    return sensor;
}

// Create an array of configuration profiles with old-style initialization
ConfigProfile* create_config_profile(void) {
    static ConfigProfile profiles[3];
    
    // Profile 0: Factory defaults
    ConfigProfile profile0 = {
        0,
        "Factory Default",
        {9600, 8, 1, 0, 0, 500},
        {"192.168.1.100", "255.255.255.0", "192.168.1.1", 8080, 1},
        {1, 100, -40, 125, 5000, 1}
    };
    profiles[0] = profile0;
    
    // Profile 1: High-speed configuration
    ConfigProfile profile1 = {
        1,
        "High Speed Mode",
        {115200, 8, 1, 1, 1, 100},
        {"10.0.0.50", "255.255.0.0", "10.0.0.1", 80, 0},
        {1, 255, -273, 1000, 1000, 1}
    };
    profiles[1] = profile1;
    
    // Profile 2: Low-power configuration
    ConfigProfile profile2 = {
        2,
        "Low Power Mode",
        {2400, 7, 2, 0, 0, 1000},
        {"172.16.0.10", "255.255.0.0", "172.16.0.1", 443, 1},
        {0, 50, 0, 85, 10000, 0}
    };
    profiles[2] = profile2;
    
    return profiles;
}

// Validate configuration values
int validate_config(const ConfigProfile* profile) {
    if (profile == NULL) {
        return 0;
    }
    
    // Validate serial parameters
    if (profile->serial.baud_rate < 300 || profile->serial.baud_rate > 115200) {
        fprintf(stderr, "Invalid baud rate: %u\n", profile->serial.baud_rate);
        return 0;
    }
    
    if (profile->serial.data_bits < 5 || profile->serial.data_bits > 8) {
        fprintf(stderr, "Invalid data bits: %u\n", profile->serial.data_bits);
        return 0;
    }
    
    if (profile->serial.stop_bits < 1 || profile->serial.stop_bits > 2) {
        fprintf(stderr, "Invalid stop bits: %u\n", profile->serial.stop_bits);
        return 0;
    }
    
    // Validate network parameters
    if (profile->network.port == 0 || profile->network.port > 65535) {
        fprintf(stderr, "Invalid port: %u\n", profile->network.port);
        return 0;
    }
    
    // Validate sensor parameters
    if (profile->sensor.min_temp > profile->sensor.max_temp) {
        fprintf(stderr, "Invalid temperature range\n");
        return 0;
    }
    
    if (profile->sensor.sample_rate == 0) {
        fprintf(stderr, "Invalid sample rate\n");
        return 0;
    }
    
    return 1;
}

// Reset configuration to factory defaults
ConfigProfile reset_to_factory(void) {
    // Factory default configuration using old positional initialization
    ConfigProfile factory = {
        0,
        "Factory Reset",
        {9600, 8, 1, 0, 0, 500},
        {"192.168.0.1", "255.255.255.0", "192.168.0.254", 80, 1},
        {1, 100, -20, 80, 2000, 1}
    };
    return factory;
}

// Load configuration from memory location
int load_from_memory(ConfigProfile* dest, const uint8_t* memory, size_t size) {
    if (dest == NULL || memory == NULL) {
        return 0;
    }
    
    if (size < sizeof(ConfigProfile)) {
        fprintf(stderr, "Insufficient memory size\n");
        return 0;
    }
    
    // Copy from memory
    memcpy(dest, memory, sizeof(ConfigProfile));
    
    // Validate loaded configuration
    if (!validate_config(dest)) {
        fprintf(stderr, "Loaded configuration is invalid\n");
        // Restore factory defaults
        *dest = reset_to_factory();
        return 0;
    }
    
    return 1;
}

// Save configuration to memory location
int save_to_memory(const ConfigProfile* src, uint8_t* memory, size_t size) {
    if (src == NULL || memory == NULL) {
        return 0;
    }
    
    if (size < sizeof(ConfigProfile)) {
        fprintf(stderr, "Insufficient memory size\n");
        return 0;
    }
    
    // Validate before saving
    if (!validate_config(src)) {
        fprintf(stderr, "Cannot save invalid configuration\n");
        return 0;
    }
    
    // Copy to memory
    memcpy(memory, src, sizeof(ConfigProfile));
    
    return 1;
}