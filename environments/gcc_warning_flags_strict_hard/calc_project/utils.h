#ifndef UTILS_H
#define UTILS_H

/* Utility function declarations for calculator project */

/* Print the result of a calculation */
void print_result(int value);

/* Get the name of an operation by its code */
char* get_operation_name(int op);

/* Validate if input string is valid */
int validate_input(char* str);

/* Display the main menu to user */
void display_menu(void);

/* Parse a string to number */
int parse_number(const char* str);

#endif