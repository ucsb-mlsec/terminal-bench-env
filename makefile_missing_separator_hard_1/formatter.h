#ifndef FORMATTER_H
#define FORMATTER_H

void format_output(double result);
void print_result(double value, char op);
void print_error(const char* message);
void print_welcome(void);

#define MAX_PRECISION 6
#define OUTPUT_WIDTH 15

#endif