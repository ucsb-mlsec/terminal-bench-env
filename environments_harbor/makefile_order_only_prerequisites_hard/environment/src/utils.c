#include <stdio.h>

/**
 * Print a message to stdout
 * @param message The message to print
 */
void print_message(const char* message) {
    if (message != NULL) {
        printf("%s\n", message);
    }
}

/**
 * Add two numbers together
 * @param a First number
 * @param b Second number
 * @return Sum of a and b
 */
int add_numbers(int a, int b) {
    return a + b;
}