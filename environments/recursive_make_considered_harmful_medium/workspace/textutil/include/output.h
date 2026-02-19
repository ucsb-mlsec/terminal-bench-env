#ifndef OUTPUT_H
#define OUTPUT_H

/*
 * Output module interface
 * Handles writing formatted text to files or stdout
 */

/* Output mode constants */
#define OUTPUT_MODE_STDOUT 0
#define OUTPUT_MODE_FILE   1

/* Write formatted text to output destination
 * Returns: 0 on success, -1 on error
 */
int write_output(const char* formatted_text);

/* Set output mode and destination file if applicable */
int set_output_mode(int mode, const char* filepath);

#endif /* OUTPUT_H */