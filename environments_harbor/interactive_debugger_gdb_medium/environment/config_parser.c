#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_SIZE 256
#define VALUE_SIZE 64

int parse_config_line(char *line) {
    char key[VALUE_SIZE];
    char value[VALUE_SIZE];
    char *equals_pos;
    
    // Remove newline if present
    line[strcspn(line, "\n")] = 0;
    
    // Skip empty lines and comments
    if (line[0] == '\0' || line[0] == '#') {
        return 0;
    }
    
    // Find the equals sign
    equals_pos = strchr(line, '=');
    if (equals_pos == NULL) {
        return 0;
    }
    
    // Split into key and value
    *equals_pos = '\0';
    char *key_part = line;
    char *value_part = equals_pos + 1;
    
    // Copy key and value - VULNERABILITY: No bounds checking!
    strcpy(key, key_part);
    strcpy(value, value_part);
    
    printf("Parsed: %s = %s\n", key, value);
    
    return 1;
}

int main(int argc, char *argv[]) {
    FILE *config_file;
    char line[MAX_LINE_SIZE];
    int line_count = 0;
    
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }
    
    config_file = fopen(argv[1], "r");
    if (config_file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", argv[1]);
        return 1;
    }
    
    while (fgets(line, MAX_LINE_SIZE, config_file) != NULL) {
        line_count++;
        parse_config_line(line);
    }
    
    fclose(config_file);
    
    printf("Configuration parsed successfully\n");
    
    return 0;
}