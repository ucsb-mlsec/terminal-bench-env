#include "../include/formatter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 80
#define INDENT_SIZE 4

/* Helper function to add indentation to a line */
static char* add_indentation(const char* line, int level) {
    int indent_chars = level * INDENT_SIZE;
    int line_len = strlen(line);
    char* indented = (char*)malloc(indent_chars + line_len + 1);
    
    if (!indented) {
        return NULL;
    }
    
    for (int i = 0; i < indent_chars; i++) {
        indented[i] = ' ';
    }
    strcpy(indented + indent_chars, line);
    
    return indented;
}

/* Helper function to wrap text at word boundaries */
static char* wrap_line(const char* text, int max_length) {
    if (strlen(text) <= max_length) {
        return strdup(text);
    }
    
    int capacity = strlen(text) * 2;
    char* result = (char*)malloc(capacity);
    if (!result) return NULL;
    
    result[0] = '\0';
    int pos = 0;
    int result_len = 0;
    
    while (pos < strlen(text)) {
        int remaining = strlen(text) - pos;
        int chunk_size = (remaining > max_length) ? max_length : remaining;
        
        if (remaining > max_length) {
            int break_point = chunk_size;
            while (break_point > 0 && text[pos + break_point] != ' ') {
                break_point--;
            }
            if (break_point > 0) {
                chunk_size = break_point;
            }
        }
        
        strncat(result, text + pos, chunk_size);
        result_len += chunk_size;
        pos += chunk_size;
        
        if (pos < strlen(text) && text[pos] == ' ') {
            pos++;
        }
        
        if (pos < strlen(text)) {
            strcat(result, "\n");
            result_len++;
        }
    }
    
    return result;
}

/* Main formatting function */
char* format_text(const ParsedData* data) {
    if (!data || !data->content) {
        return NULL;
    }
    
    char* wrapped = wrap_line(data->content, MAX_LINE_LENGTH);
    if (!wrapped) {
        return NULL;
    }
    
    char* formatted = add_indentation(wrapped, data->indent_level);
    free(wrapped);
    
    if (!formatted) {
        return NULL;
    }
    
    return formatted;
}

/* Utility function to trim whitespace from text */
char* trim_whitespace(const char* text) {
    if (!text) return NULL;
    
    while (*text == ' ' || *text == '\t' || *text == '\n') {
        text++;
    }
    
    int len = strlen(text);
    while (len > 0 && (text[len-1] == ' ' || text[len-1] == '\t' || text[len-1] == '\n')) {
        len--;
    }
    
    char* trimmed = (char*)malloc(len + 1);
    if (!trimmed) return NULL;
    
    strncpy(trimmed, text, len);
    trimmed[len] = '\0';
    
    return trimmed;
}