#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/utils.h"

#define MAX_BUFFER 256

int main(int argc, char *argv[]) {
    char input_buffer[MAX_BUFFER];
    char output_buffer[MAX_BUFFER];
    int result = 0;
    
    printf("Legacy Secure Application v1.0\n");
    printf("==============================\n\n");
    
    // Initialize buffers
    memset(input_buffer, 0, MAX_BUFFER);
    memset(output_buffer, 0, MAX_BUFFER);
    
    // Get user input
    printf("Enter data to process: ");
    if (fgets(input_buffer, MAX_BUFFER, stdin) != NULL) {
        // Remove trailing newline
        size_t len = strlen(input_buffer);
        if (len > 0 && input_buffer[len-1] == '\n') {
            input_buffer[len-1] = '\0';
        }
        
        // Process the input data
        printf("\nProcessing data...\n");
        result = process_data(input_buffer, output_buffer, MAX_BUFFER);
        
        if (result < 0) {
            fprintf(stderr, "Error: Data processing failed\n");
            return 1;
        }
        
        // Calculate results based on processed data
        int calculation = calculate_result(output_buffer);
        printf("Calculation result: %d\n", calculation);
        
        // Print summary of operations
        print_summary(input_buffer, output_buffer, calculation);
        
        printf("\nOperation completed successfully\n");
    } else {
        fprintf(stderr, "Error: Failed to read input\n");
        return 1;
    }
    
    return 0;
}