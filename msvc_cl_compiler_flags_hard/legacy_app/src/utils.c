#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "../include/utils.h"

// Calculate a simple checksum for data validation
uint32_t calculate_checksum(const uint8_t *data, size_t length) {
    uint32_t checksum = 0;
    if (data == NULL || length == 0) {
        return 0;
    }
    
    for (size_t i = 0; i < length; i++) {
        checksum += data[i];
        checksum = (checksum << 1) | (checksum >> 31);
    }
    return checksum;
}

// Format output string with prefix and timestamp
int format_output(char *buffer, size_t buffer_size, const char *prefix, const char *message) {
    if (buffer == NULL || prefix == NULL || message == NULL || buffer_size == 0) {
        return -1;
    }
    
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);
    
    int result = snprintf(buffer, buffer_size, "[%s] %s: %s", time_str, prefix, message);
    return (result >= 0 && (size_t)result < buffer_size) ? 0 : -1;
}

// Validate input string for embedded system constraints
int validate_input(const char *input, size_t max_length) {
    if (input == NULL) {
        return 0;
    }
    
    size_t length = strlen(input);
    if (length == 0 || length > max_length) {
        return 0;
    }
    
    // Check for valid printable ASCII characters
    for (size_t i = 0; i < length; i++) {
        if (input[i] < 32 || input[i] > 126) {
            return 0;
        }
    }
    return 1;
}

// Log message to stderr with severity level
void log_message(const char *level, const char *message) {
    if (level == NULL || message == NULL) {
        return;
    }
    
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", tm_info);
    
    fprintf(stderr, "[%s] %s: %s\n", time_str, level, message);
}

// Get current timestamp in milliseconds
uint64_t get_timestamp(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

// Safe string copy with size bounds checking
int safe_string_copy(char *dest, const char *src, size_t dest_size) {
    if (dest == NULL || src == NULL || dest_size == 0) {
        return -1;
    }
    
    size_t src_len = strlen(src);
    if (src_len >= dest_size) {
        return -1;
    }
    
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
    return 0;
}

// Convert integer to hexadecimal string
int int_to_hex_string(uint32_t value, char *buffer, size_t buffer_size) {
    if (buffer == NULL || buffer_size < 9) {
        return -1;
    }
    
    snprintf(buffer, buffer_size, "%08X", value);
    return 0;
}

// Memory pattern fill for testing
void memory_pattern_fill(uint8_t *buffer, size_t size, uint8_t pattern) {
    if (buffer == NULL || size == 0) {
        return;
    }
    
    for (size_t i = 0; i < size; i++) {
        buffer[i] = (uint8_t)(pattern + (i & 0xFF));
    }
}