#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math_ops.h"

void print_usage(const char* program_name) {
    printf("Usage: %s <number1> <operator> <number2>\n", program_name);
    printf("Supported operators: +, -, *, /\n");
    printf("Example: %s 10 + 5\n", program_name);
}

int main(int argc, char* argv[]) {
    double num1, num2, result;
    char* operator;

    // Check if correct number of arguments provided
    if (argc != 4) {
        fprintf(stderr, "Error: Incorrect number of arguments\n");
        print_usage(argv[0]);
        return 1;
    }

    // Parse the arguments
    num1 = atof(argv[1]);
    operator = argv[2];
    num2 = atof(argv[3]);

    // Validate operator length
    if (strlen(operator) != 1) {
        fprintf(stderr, "Error: Invalid operator '%s'\n", operator);
        print_usage(argv[0]);
        return 1;
    }

    // Perform the operation based on the operator
    switch (operator[0]) {
        case '+':
            result = add(num1, num2);
            printf("Result: %.2f\n", result);
            break;

        case '-':
            result = subtract(num1, num2);
            printf("Result: %.2f\n", result);
            break;

        case '*':
            result = multiply(num1, num2);
            printf("Result: %.2f\n", result);
            break;

        case '/':
            // Check for division by zero
            if (num2 == 0.0) {
                fprintf(stderr, "Error: Division by zero is not allowed\n");
                return 1;
            }
            result = divide(num1, num2);
            printf("Result: %.2f\n", result);
            break;

        default:
            fprintf(stderr, "Error: Unsupported operator '%s'\n", operator);
            fprintf(stderr, "Supported operators are: +, -, *, /\n");
            return 1;
    }

    return 0;
}