#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#define BUFFER_SIZE 1024

char* read_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* buffer = (char*)malloc(file_size + 1);
    if (!buffer) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    size_t bytes_read = fread(buffer, 1, file_size, file);
    buffer[bytes_read] = '\0';

    fclose(file);
    return buffer;
}

int write_file(const char* filename, const char* data) {
    if (!filename || !data) {
        return -1;
    }

    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return -1;
    }

    size_t data_len = strlen(data);
    size_t bytes_written = fwrite(data, 1, data_len, file);

    fclose(file);

    if (bytes_written != data_len) {
        fprintf(stderr, "Error: Failed to write complete data\n");
        return -1;
    }

    return 0;
}

int validate_input(const char* input) {
    if (!input) {
        return 0;
    }

    size_t len = strlen(input);
    if (len == 0 || len > 10000) {
        return 0;
    }

    for (size_t i = 0; i < len; i++) {
        if (input[i] == '\0') {
            break;
        }
    }

    return 1;
}

char* format_output(const char* input) {
    if (!input) {
        return NULL;
    }

    size_t len = strlen(input);
    char* output = (char*)malloc(len + 50);
    if (!output) {
        return NULL;
    }

    snprintf(output, len + 50, "Processed: %s", input);
    return output;
}

unsigned int calculate_checksum(const char* data) {
    if (!data) {
        return 0;
    }

    unsigned int checksum = 0;
    size_t len = strlen(data);

    for (size_t i = 0; i < len; i++) {
        checksum += (unsigned char)data[i];
        checksum = (checksum << 1) | (checksum >> 31);
    }

    return checksum;
}