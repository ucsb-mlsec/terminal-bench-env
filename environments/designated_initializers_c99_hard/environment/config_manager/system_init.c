/* system_init.c - System Initialization Module */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "config_types.h"

/* Initialize system parameters with default values */
SystemConfig init_system_params(void) {
    SystemConfig config = {57600, 8, 1, 1, 1};
    return config;
}

/* Setup memory configuration with base address and size */
MemoryConfig setup_memory_config(void) {
    MemoryConfig mem_config = {0x20000000, 0x10000, 4096, 1};
    return mem_config;
}

/* Initialize clock configuration for system timing */
ClockConfig init_clock_config(void) {
    ClockConfig clock = {48000000, 12000000, 2, 1, 0};
    return clock;
}

/* Create boot configuration with version information */
BootConfig create_boot_config(void) {
    BootConfig boot = {1, 5, 0, "bootloader_v1.0"};
    return boot;
}

/* System modules array - initialized with old-style positional syntax */
static SystemModule system_modules[] = {
    {1, "uart_driver", 1, 0, 0x1000},
    {2, "i2c_controller", 1, 0, 0x2000},
    {3, "spi_interface", 0, 0, 0x3000},
    {4, "gpio_handler", 1, 0, 0x4000}
};

/* Configure the entire system with all parameters */
int configure_system(void) {
    SystemConfig sys_cfg;
    MemoryConfig mem_cfg;
    ClockConfig clk_cfg;
    BootConfig boot_cfg;
    
    printf("Configuring system...\n");
    
    sys_cfg = init_system_params();
    printf("System params: baud=%u, data_bits=%u, stop_bits=%u\n",
           sys_cfg.baud_rate, sys_cfg.data_bits, sys_cfg.stop_bits);
    
    mem_cfg = setup_memory_config();
    printf("Memory config: base=0x%08X, size=0x%X, page_size=%u\n",
           mem_cfg.base_address, mem_cfg.size, mem_cfg.page_size);
    
    clk_cfg = init_clock_config();
    printf("Clock config: sys_freq=%u, ext_freq=%u, divider=%u\n",
           clk_cfg.system_frequency, clk_cfg.external_frequency, clk_cfg.divider);
    
    boot_cfg = create_boot_config();
    printf("Boot config: enabled=%u, timeout=%u, version=%s\n",
           boot_cfg.boot_enabled, boot_cfg.timeout_seconds, boot_cfg.version_string);
    
    return 0;
}

/* Start all enabled system modules */
int start_modules(void) {
    size_t i;
    size_t module_count = sizeof(system_modules) / sizeof(system_modules[0]);
    int started_count = 0;
    
    printf("\nStarting system modules...\n");
    
    for (i = 0; i < module_count; i++) {
        if (system_modules[i].enabled) {
            printf("Starting module %u: %s (base=0x%X)\n",
                   system_modules[i].id,
                   system_modules[i].name,
                   system_modules[i].base_register);
            system_modules[i].initialized = 1;
            started_count++;
        } else {
            printf("Module %u: %s is disabled, skipping\n",
                   system_modules[i].id,
                   system_modules[i].name);
        }
    }
    
    printf("Started %d modules successfully\n", started_count);
    return started_count;
}

/* Perform system health check on all modules */
int system_health_check(void) {
    size_t i;
    size_t module_count = sizeof(system_modules) / sizeof(system_modules[0]);
    int health_status = 0;
    int active_modules = 0;
    
    printf("\nPerforming system health check...\n");
    
    for (i = 0; i < module_count; i++) {
        if (system_modules[i].enabled && system_modules[i].initialized) {
            printf("Module %s: HEALTHY\n", system_modules[i].name);
            active_modules++;
        } else if (system_modules[i].enabled && !system_modules[i].initialized) {
            printf("Module %s: ERROR - enabled but not initialized\n",
                   system_modules[i].name);
            health_status = -1;
        }
    }
    
    printf("Health check complete: %d active modules\n", active_modules);
    
    if (health_status == 0 && active_modules > 0) {
        printf("System health: GOOD\n");
    } else if (health_status != 0) {
        printf("System health: CRITICAL\n");
    } else {
        printf("System health: WARNING - no active modules\n");
    }
    
    return health_status;
}

/* Main entry point for system initialization */
int main(void) {
    int result;
    
    printf("=== System Initialization Starting ===\n\n");
    
    result = configure_system();
    if (result != 0) {
        fprintf(stderr, "System configuration failed\n");
        return EXIT_FAILURE;
    }
    
    result = start_modules();
    if (result < 0) {
        fprintf(stderr, "Module startup failed\n");
        return EXIT_FAILURE;
    }
    
    result = system_health_check();
    if (result != 0) {
        fprintf(stderr, "Health check reported issues\n");
        return EXIT_FAILURE;
    }
    
    printf("\n=== System Initialization Complete ===\n");
    
    return EXIT_SUCCESS;
}