#include <stdint.h>
#include <string.h>

__attribute__((section(".text.util"))) uint32_t utility_crc32(const uint8_t *data, uint32_t length) {
    uint32_t crc = 0xFFFFFFFF;
    uint32_t i, j;
    
    for (i = 0; i < length; i++) {
        crc ^= data[i];
        for (j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc = crc >> 1;
            }
        }
    }
    
    return ~crc;
}

__attribute__((section(".text.util"))) void utility_memcpy_safe(void *dest, const void *src, uint32_t size, uint32_t max_size) {
    if (size > max_size) {
        size = max_size;
    }
    
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;
    
    for (uint32_t i = 0; i < size; i++) {
        d[i] = s[i];
    }
}

__attribute__((section(".text.util"))) int32_t utility_atoi(const char *str) {
    int32_t result = 0;
    int32_t sign = 1;
    uint32_t i = 0;
    
    while (str[i] == ' ' || str[i] == '\t') {
        i++;
    }
    
    if (str[i] == '-') {
        sign = -1;
        i++;
    } else if (str[i] == '+') {
        i++;
    }
    
    while (str[i] >= '0' && str[i] <= '9') {
        result = result * 10 + (str[i] - '0');
        i++;
    }
    
    return sign * result;
}

__attribute__((section(".text.util"))) uint16_t utility_swap_bytes16(uint16_t value) {
    return (value >> 8) | (value << 8);
}

__attribute__((section(".text.util"))) uint32_t utility_swap_bytes32(uint32_t value) {
    return ((value >> 24) & 0xFF) |
           ((value >> 8) & 0xFF00) |
           ((value << 8) & 0xFF0000) |
           ((value << 24) & 0xFF000000);
}