#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 256
#define MAX_KEY_LENGTH 64
#define MAX_VALUE_LENGTH 192

/* Trim whitespace from both ends of a string */
char* trim_whitespace(char* str) {
    char* end;
    
    /* Trim leading space */
    while(isspace((unsigned char)*str)) str++;
    
    if(*str == 0) return str;
    
    /* Trim trailing space */
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    
    end[1] = '\0';
    return str;
}

/* Validate a configuration line in key=value format */
int validate_config_line(char* line) {
    char* trimmed;
    char* equals_pos;
    char key[MAX_KEY_LENGTH + 1];
    char value[MAX_VALUE_LENGTH + 1];
    int key_len, value_len;
    
    /* Check line length */
    if (strlen(line) > MAX_LINE_LENGTH) {
        fprintf(stderr, "Error: Line exceeds maximum length of %d characters\n", MAX_LINE_LENGTH);
        return 0;
    }
    
    /* Trim whitespace */
    trimmed = trim_whitespace(line);
    
    /* Skip empty lines and comments */
    if (strlen(trimmed) == 0 || trimmed[0] == '#') {
        return 1;
    }
    
    /* Find the equals sign */
    equals_pos = strchr(trimmed, '=');
    if (equals_pos == NULL) {
        fprintf(stderr, "Error: Line missing '=' separator: %s\n", trimmed);
        return 0;
    }
    
    /* Extract key */
    key_len = equals_pos - trimmed;
    if (key_len == 0) {
        fprintf(stderr, "Error: Empty key in line: %s\n", trimmed);
        return 0;
    }
    
    if (key_len > MAX_KEY_LENGTH) {
        fprintf(stderr, "Error: Key exceeds maximum length of %d characters\n", MAX_KEY_LENGTH);
        return 0;
    }
    
    strncpy(key, trimmed, key_len);
    key[key_len] = '\0';
    
    /* Validate key - must contain only alphanumeric characters and underscores */
    for (int i = 0; i < key_len; i++) {
        if (!isalnum((unsigned char)key[i]) && key[i] != '_' && key[i] != '-' && key[i] != '.') {
            fprintf(stderr, "Error: Invalid character in key '%s'\n", key);
            return 0;
        }
    }
    
    /* Extract value */
    value_len = strlen(equals_pos + 1);
    if (value_len > MAX_VALUE_LENGTH) {
        fprintf(stderr, "Error: Value exceeds maximum length of %d characters\n", MAX_VALUE_LENGTH);
        return 0;
    }
    
    strncpy(value, equals_pos + 1, value_len);
    value[value_len] = '\0';
    
    /* Trim value */
    char* trimmed_value = trim_whitespace(value);
    
    /* Value can be empty, but the line must be well-formed */
    return 1;
}

int main(int argc, char* argv[]) {
    char line[MAX_LINE_LENGTH + 2]; /* +2 for newline and null terminator */
    int line_number = 0;
    int valid = 1;
    
    /* Read lines from stdin */
    while (fgets(line, sizeof(line), stdin) != NULL) {
        line_number++;
        
        /* Remove newline if present */
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }
        
        /* Validate the line */
        if (!validate_config_line(line)) {
            fprintf(stderr, "Validation failed at line %d\n", line_number);
            valid = 0;
        }
    }
    
    if (valid) {
        printf("Configuration validation successful: %d lines processed\n", line_number);
        return 0;
    } else {
        fprintf(stderr, "Configuration validation failed\n");
        return 1;
    }
}