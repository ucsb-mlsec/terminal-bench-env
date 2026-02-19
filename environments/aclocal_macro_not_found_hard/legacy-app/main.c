#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mathops.h"
#include "dbops.h"

#define VERSION "1.0.0"
#define MAX_OPERATION_LEN 20

void print_usage(const char *program_name) {
    printf("Usage: %s <operation> <num1> <num2>\n", program_name);
    printf("Operations:\n");
    printf("  add        - Addition\n");
    printf("  subtract   - Subtraction\n");
    printf("  multiply   - Multiplication\n");
    printf("  divide     - Division\n");
    printf("  power      - Power (num1^num2)\n");
    printf("\nExample: %s add 5 3\n", program_name);
    printf("Version: %s\n", VERSION);
}

int validate_operation(const char *operation) {
    if (strcmp(operation, "add") == 0) return 1;
    if (strcmp(operation, "subtract") == 0) return 1;
    if (strcmp(operation, "multiply") == 0) return 1;
    if (strcmp(operation, "divide") == 0) return 1;
    if (strcmp(operation, "power") == 0) return 1;
    return 0;
}

double perform_operation(const char *operation, double num1, double num2, int *error) {
    double result = 0.0;
    *error = 0;

    if (strcmp(operation, "add") == 0) {
        result = math_add(num1, num2);
    } else if (strcmp(operation, "subtract") == 0) {
        result = math_subtract(num1, num2);
    } else if (strcmp(operation, "multiply") == 0) {
        result = math_multiply(num1, num2);
    } else if (strcmp(operation, "divide") == 0) {
        if (num2 == 0.0) {
            fprintf(stderr, "Error: Division by zero\n");
            *error = 1;
            return 0.0;
        }
        result = math_divide(num1, num2);
    } else if (strcmp(operation, "power") == 0) {
        result = math_power(num1, num2);
    } else {
        fprintf(stderr, "Error: Unknown operation\n");
        *error = 1;
        return 0.0;
    }

    return result;
}

int main(int argc, char *argv[]) {
    char operation[MAX_OPERATION_LEN];
    double num1, num2, result;
    char *endptr;
    int error = 0;
    void *db_handle = NULL;

    // Check argument count
    if (argc != 4) {
        fprintf(stderr, "Error: Invalid number of arguments\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Copy and validate operation
    strncpy(operation, argv[1], MAX_OPERATION_LEN - 1);
    operation[MAX_OPERATION_LEN - 1] = '\0';

    if (!validate_operation(operation)) {
        fprintf(stderr, "Error: Invalid operation '%s'\n\n", operation);
        print_usage(argv[0]);
        return 1;
    }

    // Parse first number
    num1 = strtod(argv[2], &endptr);
    if (*endptr != '\0') {
        fprintf(stderr, "Error: Invalid number '%s'\n", argv[2]);
        return 1;
    }

    // Parse second number
    num2 = strtod(argv[3], &endptr);
    if (*endptr != '\0') {
        fprintf(stderr, "Error: Invalid number '%s'\n", argv[3]);
        return 1;
    }

    // Initialize database
    db_handle = db_init("operations.db");
    if (db_handle == NULL) {
        fprintf(stderr, "Warning: Could not initialize database logging\n");
    }

    // Perform the calculation
    result = perform_operation(operation, num1, num2, &error);

    if (error) {
        if (db_handle != NULL) {
            db_close(db_handle);
        }
        return 1;
    }

    // Log operation to database
    if (db_handle != NULL) {
        if (db_log_operation(db_handle, operation, num1, num2, result) != 0) {
            fprintf(stderr, "Warning: Could not log operation to database\n");
        }
    }

    // Print result
    printf("Result: %.6f %s %.6f = %.6f\n", num1, operation, num2, result);

    // Cleanup
    if (db_handle != NULL) {
        db_close(db_handle);
    }

    return 0;
}