#include "../include/parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKENS 1024
#define MAX_TOKEN_LENGTH 256

struct ParsedData* create_parsed_data(void) {
    struct ParsedData* data = (struct ParsedData*)malloc(sizeof(struct ParsedData));
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate memory for ParsedData\n");
        return NULL;
    }
    data->tokens = (char**)malloc(sizeof(char*) * MAX_TOKENS);
    if (!data->tokens) {
        fprintf(stderr, "Error: Failed to allocate memory for tokens\n");
        free(data);
        return NULL;
    }
    data->token_count = 0;
    data->capacity = MAX_TOKENS;
    return data;
}

void free_parsed_data(struct ParsedData* data) {
    if (!data) return;
    
    for (int i = 0; i < data->token_count; i++) {
        free(data->tokens[i]);
    }
    free(data->tokens);
    free(data);
}

static int add_token(struct ParsedData* data, const char* token) {
    if (data->token_count >= data->capacity) {
        fprintf(stderr, "Error: Token limit reached\n");
        return -1;
    }
    
    data->tokens[data->token_count] = strdup(token);
    if (!data->tokens[data->token_count]) {
        fprintf(stderr, "Error: Failed to duplicate token\n");
        return -1;
    }
    data->token_count++;
    return 0;
}

struct ParsedData* parse_text(const char* input) {
    if (!input) {
        fprintf(stderr, "Error: NULL input provided to parse_text\n");
        return NULL;
    }
    
    struct ParsedData* data = create_parsed_data();
    if (!data) {
        return NULL;
    }
    
    char* input_copy = strdup(input);
    if (!input_copy) {
        fprintf(stderr, "Error: Failed to copy input string\n");
        free_parsed_data(data);
        return NULL;
    }
    
    const char* delimiters = " \t\n\r,;.!?";
    char* token = strtok(input_copy, delimiters);
    
    while (token != NULL) {
        if (strlen(token) > 0) {
            if (add_token(data, token) != 0) {
                free(input_copy);
                free_parsed_data(data);
                return NULL;
            }
        }
        token = strtok(NULL, delimiters);
    }
    
    free(input_copy);
    return data;
}