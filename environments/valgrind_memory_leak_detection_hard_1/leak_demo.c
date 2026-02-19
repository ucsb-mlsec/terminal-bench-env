#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
#define MAX_WORDS 100

typedef struct {
    char *data;
    int length;
} StringBuffer;

char** tokenize_line(const char *line, int *word_count) {
    char **words = malloc(MAX_WORDS * sizeof(char*));
    *word_count = 0;
    
    // LEAK: LEAK_A
    char *line_copy = malloc(strlen(line) + 1);
    strcpy(line_copy, line);
    
    char *token = strtok(line_copy, " \t\n\r");
    while (token != NULL && *word_count < MAX_WORDS) {
        words[*word_count] = malloc(strlen(token) + 1);
        strcpy(words[*word_count], token);
        (*word_count)++;
        token = strtok(NULL, " \t\n\r");
    }
    
    return words;
}

StringBuffer* create_buffer(const char *initial) {
    StringBuffer *buffer = malloc(sizeof(StringBuffer));
    if (buffer == NULL) {
        return NULL;
    }
    
    buffer->length = strlen(initial);
    buffer->data = malloc(buffer->length + 1);
    strcpy(buffer->data, initial);
    
    return buffer;
}

void append_to_buffer(StringBuffer *buffer, const char *text) {
    int new_length = buffer->length + strlen(text);
    char *new_data = malloc(new_length + 1);
    
    strcpy(new_data, buffer->data);
    strcat(new_data, text);
    
    free(buffer->data);
    buffer->data = new_data;
    buffer->length = new_length;
}

char* process_line(const char *line) {
    int word_count;
    char **words = tokenize_line(line, &word_count);
    
    if (word_count == 0) {
        free(words);
        return NULL;
    }
    
    // LEAK: LEAK_B
    char *result = malloc(MAX_LINE_LENGTH);
    result[0] = '\0';
    
    for (int i = 0; i < word_count; i++) {
        strcat(result, words[i]);
        if (i < word_count - 1) {
            strcat(result, "_");
        }
        free(words[i]);
    }
    
    free(words);
    return result;
}

char* concatenate_strings(const char *str1, const char *str2, const char *separator) {
    int total_length = strlen(str1) + strlen(str2) + strlen(separator) + 1;
    char *result = malloc(total_length);
    
    strcpy(result, str1);
    strcat(result, separator);
    strcat(result, str2);
    
    return result;
}

int process_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return 1;
    }
    
    char line[MAX_LINE_LENGTH];
    int line_number = 0;
    StringBuffer *output = create_buffer("");
    
    printf("Processing file: %s\n", filename);
    printf("==========================================\n");
    
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL) {
        line_number++;
        
        // Remove trailing newline
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        
        if (strlen(line) == 0) {
            continue;
        }
        
        char *processed = process_line(line);
        if (processed != NULL) {
            printf("Line %d: %s\n", line_number, processed);
            
            char line_prefix[50];
            sprintf(line_prefix, "[%d] ", line_number);
            
            // LEAK: LEAK_C
            char *formatted = malloc(strlen(line_prefix) + strlen(processed) + 2);
            strcpy(formatted, line_prefix);
            strcat(formatted, processed);
            strcat(formatted, "\n");
            
            append_to_buffer(output, formatted);
            
            free(processed);
        }
    }
    
    printf("==========================================\n");
    printf("Summary:\n");
    printf("Total lines processed: %d\n", line_number);
    printf("Output buffer size: %d bytes\n", output->length);
    
    fclose(file);
    
    // Free the buffer data but not formatted strings
    free(output->data);
    free(output);
    
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    
    int result = process_file(argv[1]);
    
    if (result == 0) {
        printf("\nProcessing completed successfully.\n");
    }
    
    return result;
}