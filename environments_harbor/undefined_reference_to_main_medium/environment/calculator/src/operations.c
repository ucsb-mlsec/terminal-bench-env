#include <stdio.h>
#include <stdlib.h>
#include "operations.h"

char get_operator(char* op_str) {
    if (op_str == NULL || op_str[0] == '\0') {
        fprintf(stderr, "Error: Invalid operator\n");
        return '\0';
    }
    return op_str[0];
}

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

double calculate(double num1, char operator, double num2) {
    switch(operator) {
        case '+':
            return add(num1, num2);
        case '-':
            return subtract(num1, num2);
        case '*':
        case 'x':
        case 'X':
            return multiply(num1, num2);
        case '/':
            return divide(num1, num2);
        default:
            fprintf(stderr, "Error: Invalid operator '%c'\n", operator);
            exit(1);
    }
}

int validate_number(char* str) {
    if (str == NULL || str[0] == '\0') {
        return 0;
    }
    
    int i = 0;
    if (str[i] == '-' || str[i] == '+') {
        i++;
    }
    
    int has_digit = 0;
    int has_dot = 0;
    
    while (str[i] != '\0') {
        if (str[i] >= '0' && str[i] <= '9') {
            has_digit = 1;
        } else if (str[i] == '.' && !has_dot) {
            has_dot = 1;
        } else {
            return 0;
        }
        i++;
    }
    
    return has_digit;
}