#include <stdio.h>
#include <string.h>

// Unsafe string copy utility - uses strcpy without bounds checking
void process_string(char *dest, const char *src) {
    strcpy(dest, src);  // Buffer overflow vulnerability
}

// Unsafe formatting function - uses sprintf without size limits
void format_output(char *buffer, const char *name, int value) {
    sprintf(buffer, "Processing: %s with value %d", name, value);  // No bounds checking
}

// Unsafe concatenation - uses strcat without checking buffer size
void concat_paths(char *dest, const char *dir, const char *file) {
    strcpy(dest, dir);   // Unsafe copy
    strcat(dest, "/");   // Unsafe concatenation
    strcat(dest, file);  // Unsafe concatenation
}

// Unsafe input reading function
void read_input(char *buffer) {
    gets(buffer);  // Extremely unsafe - gets() has no bounds checking
}

// Helper function for test validation
int validate_operation(const char *input) {
    char local_buffer[64];
    char formatted[128];
    
    // Use unsafe operations
    strcpy(local_buffer, input);  // Unsafe copy
    sprintf(formatted, "Validated: %s", local_buffer);  // Unsafe format
    
    return strlen(formatted) > 0 ? 1 : 0;
}