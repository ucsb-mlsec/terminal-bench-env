#ifndef UTILS_H
#define UTILS_H

/*
 * Utility functions for data processing application
 */

/* Copy a string with specified length */
char* string_copy(const char* src, int len);

/* Parse a string to integer with error handling */
int parse_number(const char* str);

/* Validate input data format */
int validate_input(const char* input);

/* Format output data into a buffer */
void format_output(char* buffer, int size, const char* data);

#endif /* UTILS_H */