#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "processor.h"
#include "utils.h"

#define VERSION "1.0.0"

void print_usage(const char *progname) {
    printf("Usage: %s <input_file>\n", progname);
    printf("Data Processing Application v%s\n", VERSION);
    printf("Processes data from the specified input file.\n");
}

int main(int argc, char *argv[]) {
    int result = 0;
    char *input_file = NULL;
    char *file_content = NULL;
    long file_size = 0;

    printf("Data Processor v%s starting...\n", VERSION);

    // Check command-line arguments
    if (argc < 2) {
        fprintf(stderr, "Error: No input file specified\n");
        print_usage(argv[0]);
        return 1;
    }

    input_file = argv[1];
    printf("Input file: %s\n", input_file);

    // Validate input file
    if (!validate_input(input_file)) {
        fprintf(stderr, "Error: Invalid input file: %s\n", input_file);
        return 2;
    }

    printf("Validating input... OK\n");

    // Read file content
    file_content = read_file(input_file, &file_size);
    if (file_content == NULL) {
        fprintf(stderr, "Error: Failed to read file: %s\n", input_file);
        return 3;
    }

    printf("File size: %ld bytes\n", file_size);

    // Initialize processor
    if (!init_processor()) {
        fprintf(stderr, "Error: Failed to initialize processor\n");
        free(file_content);
        return 4;
    }

    printf("Processor initialized\n");

    // Process data
    result = process_data(file_content, file_size);
    if (result < 0) {
        fprintf(stderr, "Error: Data processing failed with code %d\n", result);
        cleanup_processor();
        free(file_content);
        return 5;
    }

    printf("Processing completed successfully\n");
    printf("Records processed: %d\n", result);

    // Cleanup
    cleanup_processor();
    free(file_content);

    printf("Data Processor finished\n");
    return 0;
}