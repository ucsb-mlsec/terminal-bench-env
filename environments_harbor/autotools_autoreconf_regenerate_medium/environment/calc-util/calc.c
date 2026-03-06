#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int num1, num2, result;
    char *operation;

    // Check if correct number of arguments provided
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <number1> <number2> <operation>\n", argv[0]);
        fprintf(stderr, "Operations: add, subtract, multiply, divide\n");
        return 1;
    }

    // Parse arguments
    num1 = atoi(argv[1]);
    num2 = atoi(argv[2]);
    operation = argv[3];

    // Perform the requested operation
    if (strcmp(operation, "add") == 0) {
        result = num1 + num2;
        printf("%d\n", result);
    }
    else if (strcmp(operation, "subtract") == 0) {
        result = num1 - num2;
        printf("%d\n", result);
    }
    else if (strcmp(operation, "multiply") == 0) {
        result = num1 * num2;
        printf("%d\n", result);
    }
    else if (strcmp(operation, "divide") == 0) {
        if (num2 == 0) {
            fprintf(stderr, "Error: Division by zero\n");
            return 1;
        }
        result = num1 / num2;
        printf("%d\n", result);
    }
    else {
        fprintf(stderr, "Error: Unknown operation '%s'\n", operation);
        fprintf(stderr, "Supported operations: add, subtract, multiply, divide\n");
        return 1;
    }

    return 0;
}