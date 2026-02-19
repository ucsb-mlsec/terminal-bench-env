#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

char* string_copy(const char* source) {
    if (source == NULL) {
        return NULL;
    }
    
    size_t len = strlen(source);
    char* dest = (char*)malloc(len); // Off-by-one: missing +1 for null terminator
    
    if (dest == NULL) {
        return NULL;
    }
    
    strcpy(dest, source);
    return dest;
}

int parse_number(const char* str) {
    if (str == NULL) {
        return -1;
    }
    
    int result = 0;
    int sign = 1;
    int i = 0;
    
    if (str[0] == '-') {
        sign = -1;
        i = 1;
    }
    
    while (str[i] >= '0' && str[i] <= '9') {
        result = result * 10 + (str[i] - '0');
        i++;
    }
    
    return result * sign;
}

int validate_input(const char* input, int min_len, int max_len) {
    if (input == NULL) {
        return 0;
    }
    
    size_t len = strlen(input);
    
    if (len < min_len || len > max_len) {
        return 0;
    }
    
    for (size_t i = 0; i < len; i++) {
        if (input[i] < 32 || input[i] > 126) {
            return 0;
        }
    }
    
    return 1;
}

char* format_output(const char* prefix, const char* data, const char* suffix) {
    if (prefix == NULL || data == NULL || suffix == NULL) {
        return NULL;
    }
    
    size_t total_len = strlen(prefix) + strlen(data) + strlen(suffix) + 1;
    char* output = (char*)malloc(total_len);
    
    if (output == NULL) {
        return NULL;
    }
    
    strcpy(output, prefix);
    strcat(output, data);
    strcat(output, suffix);
    
    return output;
}