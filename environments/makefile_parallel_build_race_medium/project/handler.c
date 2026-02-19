#include <stdio.h>
#include "handler.h"
#include "generated.h"

int handle_request(int request_id) {
    printf("Processing request: %d\n", request_id);
    
    if (request_id < 0 || request_id >= MAX_ITEMS) {
        printf("Error: Request ID out of range (max: %d)\n", MAX_ITEMS);
        return -1;
    }
    
    printf("Request validated against MAX_ITEMS=%d\n", MAX_ITEMS);
    
    for (int i = 0; i < TIMEOUT_VALUE; i++) {
        // Simulate some processing work
        if (i == request_id) {
            printf("Found matching item at index %d\n", i);
            break;
        }
    }
    
    return 0;
}