#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include "processor.h"

#define BUFFER_SIZE 4096
#define SUCCESS 0
#define ERROR_FILE_NOT_FOUND -1
#define ERROR_INVALID_INPUT -2

/* File I/O functions */
char* read_file(const char* filename, size_t* size);
int write_file(const char* filename, const char* data, size_t size);

/* Validation and formatting */
int validate_input(const char* data, size_t size);
char* format_output(const ProcessedData* data);

/* Checksum calculation */
unsigned long calculate_checksum(const char* data, size_t size);

/* Helper function to clean up resources */
void cleanup_buffer(char* buffer);

#endif /* UTILS_H */