#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// External function from utils.c
extern void process_input(char *input);

void test_basic_operations() {
    char buffer[50];
    char dest[20];
    char output[100];
    char input_buf[30];
    
    // Unsafe strcpy - should use strncpy
    strcpy(dest, "test string that is safe");
    
    // Unsafe sprintf - should use snprintf
    sprintf(output, "Testing output: %s", dest);
    
    // Simulate unsafe gets usage - should use fgets
    // Note: gets is removed in C11, but similar unsafe pattern
    char temp[15];
    strcpy(temp, "simulated");
}

void test_string_operations() {
    char buf1[10];
    char buf2[50];
    
    // Another unsafe strcpy
    strcpy(buf1, "short");
    
    // Unsafe sprintf with formatting
    sprintf(buf2, "Value: %d, String: %s", 42, buf1);
}

int run_tests() {
    test_basic_operations();
    test_string_operations();
    return 0;
}

int main(int argc, char *argv[]) {
    char buffer[100];
    char command[64];
    
    if (argc > 1 && strcmp(argv[1], "--test") == 0) {
        // Run in test mode
        if (run_tests() == 0) {
            printf("SUCCESS\n");
            return 0;
        } else {
            printf("FAILED\n");
            return 1;
        }
    }
    
    // Normal operation mode with unsafe operations
    if (argc > 1) {
        // Unsafe strcpy from command line argument
        strcpy(command, argv[1]);
        
        // Unsafe sprintf
        sprintf(buffer, "Processing command: %s", command);
        
        process_input(buffer);
    } else {
        printf("Usage: %s [command] or %s --test\n", argv[0], argv[0]);
    }
    
    return 0;
}