#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "text_buffer.h"

#define SMALL_OPS 10000
#define BULK_OPS 1000

void run_small_insertions(TextBuffer* buffer) {
    // Simulate realistic text editor usage with insertions at various positions
    for (int i = 0; i < SMALL_OPS; i++) {
        // Insert at beginning (worst case for array-based implementation)
        if (i % 3 == 0) {
            text_buffer_insert(buffer, 0, "text ");
        }
        // Insert at middle
        else if (i % 3 == 1) {
            int pos = text_buffer_length(buffer) / 2;
            text_buffer_insert(buffer, pos, "code ");
        }
        // Insert at end (best case)
        else {
            int pos = text_buffer_length(buffer);
            text_buffer_insert(buffer, pos, "data ");
        }
    }
}

void run_bulk_operations(TextBuffer* buffer) {
    // Perform larger insertions at beginning (stress test)
    char bulk_text[100];
    strcpy(bulk_text, "This is a longer line of text that represents a typical editor operation. ");
    
    for (int i = 0; i < BULK_OPS; i++) {
        // Most insertions at beginning to expose O(n) vs O(1) difference
        text_buffer_insert(buffer, 0, bulk_text);
        
        // Some deletions
        if (i % 10 == 0 && text_buffer_length(buffer) > 100) {
            text_buffer_delete(buffer, 50, 50);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [A|B]\n", argv[0]);
        return 1;
    }
    
    char implementation = argv[1][0];
    if (implementation != 'A' && implementation != 'B') {
        fprintf(stderr, "Implementation must be A or B\n");
        return 1;
    }
    
    printf("Starting performance test for Implementation %c\n", implementation);
    printf("Running %d small operations and %d bulk operations...\n", SMALL_OPS, BULK_OPS);
    fflush(stdout);
    
    TextBuffer* buffer = text_buffer_create();
    if (!buffer) {
        fprintf(stderr, "Failed to create text buffer\n");
        return 1;
    }
    
    clock_t start = clock();
    
    // Run the performance tests
    run_small_insertions(buffer);
    run_bulk_operations(buffer);
    
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("Implementation %c completed in %.2f seconds\n", implementation, elapsed);
    printf("Final buffer size: %d characters\n", text_buffer_length(buffer));
    
    text_buffer_destroy(buffer);
    
    return 0;
}