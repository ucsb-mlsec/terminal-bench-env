#include <stdio.h>
#include <stdlib.h>
#include "parser.h"
#include "formatter.h"

double add(double a, double b) {
    return a + b;
}

double subtract(double a, double b) {
    return a - b;
}

double multiply(double a, double b) {
    return a * b;
}

double divide(double a, double b) {
    if (b == 0) {
        fprintf(stderr, "Error: Division by zero\n");
        return 0;
    }
    return a / b;
}

int main(int argc, char *argv[]) {
    double num1, num2, result;
    char operator;
    
    printf("Simple Calculator\n");
    printf("=================\n\n");
    
    if (argc > 1) {
        if (parse_input(argv[1], &num1, &operator, &num2) != 0) {
            fprintf(stderr, "Error: Invalid input format\n");
            fprintf(stderr, "Usage: %s \"num1 operator num2\"\n", argv[0]);
            fprintf(stderr, "Example: %s \"5 + 3\"\n", argv[0]);
            return 1;
        }
    } else {
        printf("Enter calculation (e.g., 5 + 3): ");
        char input[256];
        if (fgets(input, sizeof(input), stdin) == NULL) {
            fprintf(stderr, "Error reading input\n");
            return 1;
        }
        
        if (parse_input(input, &num1, &operator, &num2) != 0) {
            fprintf(stderr, "Error: Invalid input format\n");
            return 1;
        }
    }
    
    switch (operator) {
        case '+':
            result = add(num1, num2);
            break;
        case '-':
            result = subtract(num1, num2);
            break;
        case '*':
        case 'x':
            result = multiply(num1, num2);
            break;
        case '/':
            result = divide(num1, num2);
            break;
        default:
            fprintf(stderr, "Error: Unknown operator '%c'\n", operator);
            return 1;
    }
    
    format_output(num1, operator, num2, result);
    
    return 0;
}