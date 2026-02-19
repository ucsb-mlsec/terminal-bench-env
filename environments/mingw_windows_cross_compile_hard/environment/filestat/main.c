#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "processor.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        fprintf(stderr, "Process a text file and display statistics.\n");
        return 1;
    }

    const char *filename = argv[1];
    
    printf("Processing file: %s\n", filename);
    
    int result = process_file(filename);
    
    if (result == 0) {
        printf("\nFile processed successfully.\n");
        return 0;
    } else {
        fprintf(stderr, "\nError: Failed to process file '%s'\n", filename);
        return 1;
    }
}