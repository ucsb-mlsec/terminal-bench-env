#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// External function declarations
int add(int a, int b);
int subtract(int a, int b);
int multiply(int a, int b);
int divide(int a, int b);

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <operand1> <operator> <operand2>\n", argv[0]);
        fprintf(stderr, "Example: %s 10 + 5\n", argv[0]);
        return 1;
    }

    int operand1 = atoi(argv[1]);
    int operand2 = atoi(argv[3]);
    char operator = argv[2][0];
    int result;

    switch (operator) {
        case '+':
            result = add(operand1, operand2);
            break;
        case '-':
            result = subtract(operand1, operand2);
            break;
        case '*':
            result = multiply(operand1, operand2);
            break;
        case '/':
            if (operand2 == 0) {
                fprintf(stderr, "Error: Division by zero\n");
                return 1;
            }
            result = divide(operand1, operand2);
            break;
        default:
            fprintf(stderr, "Error: Unknown operator '%c'\n", operator);
            fprintf(stderr, "Supported operators: +, -, *, /\n");
            return 1;
    }

    printf("%d\n", result);
    return 0;
}