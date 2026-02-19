#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Constants */
#define MAX_LOG_LENGTH 256
#define MAX_TIMESTAMP_LENGTH 32
#define MAX_OUTPUT_LENGTH 512

/* Checksum calculation */
uint32_t calculate_checksum(const uint8_t *data, size_t length);

/* Output formatting */
int format_output(char *buffer, size_t buffer_size, 
                  const char *format, const void *data);

/* Input validation */
bool validate_input(const uint8_t *input, size_t length, 
                    uint32_t expected_checksum);

/* Logging functionality */
void log_message(const char *level, const char *message);

/* Timestamp utilities */
int get_timestamp(char *buffer, size_t buffer_size);

/* Helper macros */
#define LOG_ERROR(msg) log_message("ERROR", msg)
#define LOG_INFO(msg) log_message("INFO", msg)
#define LOG_DEBUG(msg) log_message("DEBUG", msg)

#endif /* UTILS_H */