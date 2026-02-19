#include <stdio.h>
#include <stdlib.h>
#include "generated.h"
#include "utils.h"
#include "processor.h"

int main(void) {
    printf("Application starting...\n");
    printf("Version: %d\n", VERSION_NUMBER);
    printf("Max buffer size: %d\n", MAX_BUFFER_SIZE);
    
    int result = utility_func(MAX_BUFFER_SIZE);
    if (result != 0) {
        fprintf(stderr, "Utility function failed\n");
        return 1;
    }
    
    int data[MAX_BUFFER_SIZE];
    for (int i = 0; i < MAX_BUFFER_SIZE; i++) {
        data[i] = i * 2;
    }
    
    process_data(data, MAX_BUFFER_SIZE);
    
    printf("Processing complete. Build version: %d\n", VERSION_NUMBER);
    printf("Application finished successfully.\n");
    
    return 0;
}