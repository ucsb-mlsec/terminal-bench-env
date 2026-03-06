#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Global variables - some used, some unused
uint32_t g_debug_counter = 0;
uint32_t g_error_count = 0;
static uint8_t g_unused_buffer[256];
const char *g_version_string = "v1.2.3";

// Function used by main - byte swapping for endianness
uint16_t swap_bytes_16(uint16_t value) {
    return (value >> 8) | (value << 8);
}

// Function used by main - simple checksum calculation
uint8_t calculate_checksum(const uint8_t *data, size_t length) {
    uint8_t checksum = 0;
    for (size_t i = 0; i < length; i++) {
        checksum ^= data[i];
    }
    return checksum;
}

// Function used by main - validate data range
int validate_range(int value, int min, int max) {
    return (value >= min && value <= max) ? 1 : 0;
}

// DEAD CODE: Never called - CRC16 calculation
static uint16_t calculate_crc16(const uint8_t *data, size_t length) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc = crc >> 1;
            }
        }
    }
    return crc;
}

// DEAD CODE: Never called - complex string formatting
static char *format_hex_dump(const uint8_t *data, size_t length) {
    char *buffer = malloc(length * 3 + 1);
    if (!buffer) return NULL;
    
    for (size_t i = 0; i < length; i++) {
        sprintf(buffer + (i * 3), "%02X ", data[i]);
    }
    buffer[length * 3] = '\0';
    return buffer;
}

// DEAD CODE: Never called - polynomial hash function
uint32_t polynomial_hash(const char *str) {
    uint32_t hash = 0;
    const uint32_t prime = 31;
    
    while (*str) {
        hash = hash * prime + (uint8_t)(*str);
        str++;
    }
    return hash;
}

// DEAD CODE: Never called - bit counting
static int count_set_bits(uint32_t value) {
    int count = 0;
    while (value) {
        count += value & 1;
        value >>= 1;
    }
    return count;
}

// Function used by main - simple string copy with length limit
void safe_string_copy(char *dest, const char *src, size_t max_len) {
    strncpy(dest, src, max_len - 1);
    dest[max_len - 1] = '\0';
}

// DEAD CODE: Never called - rotate bits left
uint32_t rotate_left(uint32_t value, int shift) {
    shift &= 31;
    return (value << shift) | (value >> (32 - shift));
}

// DEAD CODE: Never called - decode base64
static int decode_base64(const char *input, uint8_t *output, size_t *output_len) {
    const char *base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t input_len = strlen(input);
    size_t out_idx = 0;
    
    for (size_t i = 0; i < input_len; i += 4) {
        uint32_t triple = 0;
        for (int j = 0; j < 4; j++) {
            if (i + j < input_len) {
                const char *p = strchr(base64_chars, input[i + j]);
                if (p) {
                    triple = (triple << 6) | (p - base64_chars);
                }
            }
        }
        
        if (out_idx < *output_len) output[out_idx++] = (triple >> 16) & 0xFF;
        if (out_idx < *output_len) output[out_idx++] = (triple >> 8) & 0xFF;
        if (out_idx < *output_len) output[out_idx++] = triple & 0xFF;
    }
    
    *output_len = out_idx;
    return 0;
}

// DEAD CODE: Never called - binary search
static int binary_search(const int *array, int size, int target) {
    int left = 0;
    int right = size - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (array[mid] == target) {
            return mid;
        } else if (array[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}