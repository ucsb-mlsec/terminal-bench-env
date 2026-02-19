#include <stdint.h>

// Infrequently-called initialization and diagnostic routines
// These functions are placed in .text.slow section for memory optimization

__attribute__((section(".text.slow")))
void slow_operation(uint32_t delay_ms) {
    // Simulate a slow blocking operation
    volatile uint32_t counter = 0;
    for (uint32_t i = 0; i < delay_ms * 1000; i++) {
        counter++;
        if (counter % 1000 == 0) {
            // Prevent optimization
            counter = counter & 0xFFFFFFFF;
        }
    }
}

__attribute__((section(".text.slow")))
int initialization_task(void) {
    // Perform system initialization tasks
    // These are called once at startup
    uint32_t status = 0;
    
    // Initialize peripheral registers
    volatile uint32_t *peripheral_base = (volatile uint32_t *)0x40000000;
    peripheral_base[0] = 0x00000001;  // Enable clock
    peripheral_base[1] = 0x00000003;  // Configure mode
    peripheral_base[2] = 0x000000FF;  // Set defaults
    
    // Verify initialization
    if (peripheral_base[0] & 0x01) {
        status |= 0x01;
    }
    
    // Clear any pending flags
    peripheral_base[3] = 0xFFFFFFFF;
    
    return (status == 0x01) ? 0 : -1;
}

__attribute__((section(".text.slow")))
uint32_t diagnostic_check(void) {
    // Run comprehensive system diagnostics
    // This is typically called during maintenance or error conditions
    uint32_t error_flags = 0;
    
    // Check memory regions
    volatile uint32_t *ram_test = (volatile uint32_t *)0x20000000;
    uint32_t original_value = *ram_test;
    *ram_test = 0xDEADBEEF;
    if (*ram_test != 0xDEADBEEF) {
        error_flags |= 0x01;  // RAM test failed
    }
    *ram_test = original_value;
    
    // Check peripheral status
    volatile uint32_t *status_reg = (volatile uint32_t *)0x40000010;
    uint32_t status = *status_reg;
    if (status & 0x80000000) {
        error_flags |= 0x02;  // Error flag set
    }
    
    // Verify watchdog configuration
    if ((status & 0x0000FF00) == 0) {
        error_flags |= 0x04;  // Watchdog not configured
    }
    
    return error_flags;
}