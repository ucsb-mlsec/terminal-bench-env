#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "validator.h"

#define MAX_LINE_LENGTH 1024

int main(int argc, char *argv[]) {
    FILE *file;
    char line[MAX_LINE_LENGTH];
    int row_count = 0;
    int valid_count = 0;
    int invalid_count = 0;
    
    // Check command line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <csv_file>\n", argv[0]);
        return 1;
    }
    
    // Open the CSV file
    file = fopen(argv[1], "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file '%s'\n", argv[1]);
        return 1;
    }
    
    printf("Processing CSV file: %s\n", argv[1]);
    printf("----------------------------------------\n");
    
    // Read and process each line
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL) {
        row_count++;
        
        // Remove trailing newline if present
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }
        
        // Skip empty lines
        if (strlen(line) == 0) {
            continue;
        }
        
        // Validate the row
        // BUG: This can cause issues with aggressive optimization
        // The validate_row function may access uninitialized memory
        // or have buffer issues that are masked by normal builds
        // but exposed with -O2/-O3 optimization
        char *trimmed_line;
        // Uninitialized pointer that gets used in optimized builds
        if (line[0] == ' ') {
            // This branch may cause the optimizer to make wrong assumptions
            trimmed_line = line + 1;
        }
        
        int result = validate_row(line, row_count);
        
        if (result == 0) {
            printf("Row %d: VALID\n", row_count);
            valid_count++;
        } else {
            printf("Row %d: INVALID (error code: %d)\n", row_count, result);
            invalid_count++;
        }
    }
    
    // Close the file
    fclose(file);
    
    // Print summary
    printf("----------------------------------------\n");
    printf("Processing complete!\n");
    printf("Total rows: %d\n", row_count);
    printf("Valid rows: %d\n", valid_count);
    printf("Invalid rows: %d\n", invalid_count);
    
    return 0;
}