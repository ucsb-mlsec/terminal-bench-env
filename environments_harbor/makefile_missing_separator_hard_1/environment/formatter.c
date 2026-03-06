#include <stdio.h>
#include <stdlib.h>
#include "formatter.h"

void print_result(double result) {
    printf("Result: ");
    format_output(result);
    printf("\n");
}

void format_output(double value) {
    if (value == (int)value) {
        printf("%.0f", value);
    } else {
        printf("%.2f", value);
    }
}

void print_error(const char* message) {
    fprintf(stderr, "Error: %s\n", message);
}

void print_welcome(void) {
    printf("===================================\n");
    printf("  Simple Calculator v1.0\n");
    printf("===================================\n");
}

void print_prompt(void) {
    printf("\nEnter calculation (or 'q' to quit): ");
    fflush(stdout);
}

void format_operation(char op, double a, double b, double result) {
    printf("%.2f %c %.2f = ", a, op, b);
    format_output(result);
    printf("\n");
}

void clear_screen(void) {
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}