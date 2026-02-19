#include <stdint.h>

// External function declarations from other modules
extern void fast_operation(uint32_t value);
extern void slow_operation(void);
extern uint32_t utility_function(uint32_t a, uint32_t b);

// Performance-critical initialization function in .text.fast section
__attribute__((section(".text.fast")))
void critical_init(void) {
    volatile uint32_t *control_reg = (volatile uint32_t *)0x40000000;
    *control_reg = 0x00000001;
    
    // Time-critical configuration
    for (int i = 0; i < 10; i++) {
        control_reg[i] = i * 2;
    }
}

// Another fast function for critical path operations
__attribute__((section(".text.fast")))
uint32_t fast_compute(uint32_t input) {
    uint32_t result = input;
    result = (result << 3) | (result >> 29);
    result ^= 0xDEADBEEF;
    return result;
}

// Main firmware entry point
int main(void) {
    uint32_t system_state = 0;
    uint32_t processed_value = 0;
    
    // Critical initialization sequence
    critical_init();
    
    // Perform fast operation on initial value
    system_state = fast_compute(0x12345678);
    fast_operation(system_state);
    
    // Execute utility functions for data processing
    processed_value = utility_function(system_state, 0xABCD);
    processed_value = utility_function(processed_value, 0x1234);
    
    // Slow operations for non-critical tasks
    slow_operation();
    
    // Main firmware loop
    while (1) {
        system_state = fast_compute(system_state);
        
        if (system_state & 0x100) {
            fast_operation(system_state);
        }
        
        if ((system_state & 0xFF) == 0) {
            slow_operation();
        }
    }
    
    return 0;
}