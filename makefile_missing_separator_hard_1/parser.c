#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "parser.h"

int parse_operator(const char *str, char *op) {
    if (str == NULL || op == NULL) {
        return -1;
    }
    
    if (strlen(str) != 1) {
        return -1;
    }
    
    char c = str[0];
    if (c == '+' || c == '-' || c == '*' || c == '/') {
        *op = c;
        return 0;
    }
    
    return -1;
}

int parse_number(const char *str, double *num) {
    if (str == NULL || num == NULL) {
        return -1;
    }
    
    char *endptr;
    *num = strtod(str, &endptr);
    
    if (endptr == str || *endptr != '\0') {
        return -1;
    }
    
    return 0;
}

int parse_input(const char *input, double *num1, char *op, double *num2) {
    if (input == NULL || num1 == NULL || op == NULL || num2 == NULL) {
        return -1;
    }
    
    char buffer[256];
    strncpy(buffer, input, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';
    
    char *token1 = strtok(buffer, " \t\n");
    char *token_op = strtok(NULL, " \t\n");
    char *token2 = strtok(NULL, " \t\n");
    
    if (token1 == NULL || token_op == NULL || token2 == NULL) {
        return -1;
    }
    
    if (parse_number(token1, num1) != 0) {
        return -1;
    }
    
    if (parse_operator(token_op, op) != 0) {
        return -1;
    }
    
    if (parse_number(token2, num2) != 0) {
        return -1;
    }
    
    return 0;
}