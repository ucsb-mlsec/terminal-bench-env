#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

/* Function declarations for utility operations */

/**
 * Process input data and perform validation
 * @param input: pointer to input string
 * @return: 0 on success, -1 on failure
 */
int process_data(char* input);

/**
 * Calculate result from two integer inputs
 * @param x: first operand
 * @param y: second operand
 * @return: calculated result
 */
int calculate_result(int x, int y);

/**
 * Print summary of operations
 */
void print_summary(void);

/**
 * Initialize application resources
 * @return: 0 on success, -1 on failure
 */
int initialize_app(void);

/**
 * Cleanup and release resources
 */
void cleanup_app(void);

#endif /* UTILS_H */