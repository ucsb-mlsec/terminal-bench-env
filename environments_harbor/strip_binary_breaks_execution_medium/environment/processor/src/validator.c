#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "validator.h"

#define MAX_FIELD_LENGTH 256
#define MIN_FIELD_COUNT 3

/* Helper function to extract CSV fields */
static int parse_csv_field(const char *row, int field_index, char *output) {
    int current_field = 0;
    int output_pos = 0;
    const char *ptr = row;
    
    while (*ptr != '\0' && current_field < field_index) {
        if (*ptr == ',') {
            current_field++;
        }
        ptr++;
    }
    
    if (current_field != field_index) {
        return -1;
    }
    
    /* Extract the field - BUG: no bounds checking on output buffer */
    while (*ptr != '\0' && *ptr != ',' && *ptr != '\n') {
        output[output_pos++] = *ptr++;
    }
    output[output_pos] = '\0';
    
    return output_pos;
}

/* Validate a single CSV row */
int validate_row(char *row) {
    if (row == NULL) {
        return -1;
    }
    
    /* Remove trailing newline if present */
    size_t len = strlen(row);
    if (len > 0 && row[len - 1] == '\n') {
        row[len - 1] = '\0';
        len--;
    }
    
    /* Check for empty row */
    if (len == 0) {
        return -1;
    }
    
    /* Count fields by counting commas */
    int field_count = 1;
    for (size_t i = 0; i < len; i++) {
        if (row[i] == ',') {
            field_count++;
        }
    }
    
    /* Check minimum field count */
    if (field_count < MIN_FIELD_COUNT) {
        return -1;
    }
    
    /* Validate each field is non-empty */
    char field_buffer[MAX_FIELD_LENGTH];
    for (int i = 0; i < field_count; i++) {
        int field_len = parse_csv_field(row, i, field_buffer);
        
        if (field_len < 0) {
            return -1;
        }
        
        /* Check for empty field */
        if (field_len == 0) {
            return -1;
        }
        
        /* Trim whitespace and check again */
        int has_content = 0;
        for (int j = 0; j < field_len; j++) {
            if (!isspace((unsigned char)field_buffer[j])) {
                has_content = 1;
                break;
            }
        }
        
        if (!has_content) {
            return -1;
        }
    }
    
    return 0;
}

/* Additional validation for data types in specific columns */
int validate_data_types(char *row) {
    if (row == NULL) {
        return -1;
    }
    
    char field[MAX_FIELD_LENGTH];
    
    /* Validate first field (should be numeric ID) */
    if (parse_csv_field(row, 0, field) > 0) {
        for (size_t i = 0; i < strlen(field); i++) {
            if (!isdigit((unsigned char)field[i])) {
                return -1;
            }
        }
    }
    
    return 0;
}