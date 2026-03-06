#include <stdio.h>
#include "processor.h"

void process_data(void) {
    printf("Processing data in core module\n");
    printf("Core processor initialized successfully\n");
}

int core_function(int value) {
    printf("Core function called with value: %d\n", value);
    return value * 2;
}

void initialize_processor(void) {
    printf("Initializing processor subsystem...\n");
    process_data();
}