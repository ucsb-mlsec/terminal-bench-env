#ifndef PARSER_H
#define PARSER_H

#define MAX_INPUT 256

typedef struct {
    double operand1;
    double operand2;
    char operator;
} ParsedInput;

int parse_input(char* input, ParsedInput* result);
double parse_number(char* str);
int validate_operator(char op);

#endif