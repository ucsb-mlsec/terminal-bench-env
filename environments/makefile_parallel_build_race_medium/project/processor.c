#include <stdio.h>
#include <stdlib.h>
#include "processor.h"
#include "generated.h"

int process_data(void) {
    printf("Starting data processor...\n");
    printf("Using buffer size: %d\n", BUFFER_SIZE);
    
    char buffer[BUFFER_SIZE];
    
    for (int i = 0; i < BUFFER_SIZE && i < 10; i++) {
        buffer[i] = 'A' + i;
    }
    
    printf("Processed %d bytes\n", BUFFER_SIZE);
    printf("Configuration value: %d\n", CONFIG_VALUE);
    
    if (CONFIG_VALUE > 0) {
        printf("Configuration is valid\n");
    } else {
        printf("Using default configuration\n");
    }
    
    printf("Data processing complete\n");
    
    return 0;
}