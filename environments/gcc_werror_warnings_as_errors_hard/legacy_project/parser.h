#ifndef PARSER_H
#define PARSER_H

#include <stdio.h>
#include <stdlib.h>

int parse_config_file(const char *filename);
char* parse_line(const char *input, int max_len);
void parse_arguments(int argc, char **argv);
int validate_token(const char *token, unsigned int length);

#endif