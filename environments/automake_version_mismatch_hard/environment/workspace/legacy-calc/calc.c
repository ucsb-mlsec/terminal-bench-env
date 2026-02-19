#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple calculator program */

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
        exit(1);
    }
    return a / b;
}

void print_usage(const char *progname) {
    printf("Usage: %s <num1> <operation> <num2>\n", progname);
    printf("Operations: +, -, *, /\n");
    printf("Example: %s 10 + 5\n", progname);
}

int main(int argc, char *argv[]) {
    double num1, num2, result;
    char operation;

    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    /* Parse first number */
    num1 = atof(argv[1]);
    
    /* Parse operation */
    if (strlen(argv[2]) != 1) {
        fprintf(stderr, "Error: Operation must be a single character (+, -, *, /)\n");
        print_usage(argv[0]);
        return 1;
    }
    operation = argv[2][0];
    
    /* Parse second number */
    num2 = atof(argv[3]);

    /* Perform calculation */
    switch (operation) {
        case '+':
            result = add(num1, num2);
            printf("%.2f + %.2f = %.2f\n", num1, num2, result);
            break;
        case '-':
            result = subtract(num1, num2);
            printf("%.2f - %.2f = %.2f\n", num1, num2, result);
            break;
        case '*':
        case 'x':
        case 'X':
            result = multiply(num1, num2);
            printf("%.2f * %.2f = %.2f\n", num1, num2, result);
            break;
        case '/':
            result = divide(num1, num2);
            printf("%.2f / %.2f = %.2f\n", num1, num2, result);
            break;
        default:
            fprintf(stderr, "Error: Unknown operation '%c'\n", operation);
            print_usage(argv[0]);
            return 1;
    }

    return 0;
}