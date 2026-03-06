#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_usage(const char *program_name) {
    printf("Usage: %s <number1> <operator> <number2>\n", program_name);
    printf("Operators: +, -, *, /\n");
    printf("Example: %s 10 + 5\n", program_name);
}

int main(int argc, char *argv[]) {
    double num1, num2, result;
    char operator;

    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    num1 = atof(argv[1]);
    operator = argv[2][0];
    num2 = atof(argv[3]);

    switch(operator) {
        case '+':
            result = num1 + num2;
            printf("%.2f + %.2f = %.2f\n", num1, num2, result);
            break;
        case '-':
            result = num1 - num2;
            printf("%.2f - %.2f = %.2f\n", num1, num2, result);
            break;
        case '*':
        case 'x':
        case 'X':
            result = num1 * num2;
            printf("%.2f * %.2f = %.2f\n", num1, num2, result);
            break;
        case '/':
            if (num2 == 0) {
                fprintf(stderr, "Error: Division by zero\n");
                return 1;
            }
            result = num1 / num2;
            printf("%.2f / %.2f = %.2f\n", num1, num2, result);
            break;
        default:
            fprintf(stderr, "Error: Unknown operator '%c'\n", operator);
            print_usage(argv[0]);
            return 1;
    }

    return 0;
}