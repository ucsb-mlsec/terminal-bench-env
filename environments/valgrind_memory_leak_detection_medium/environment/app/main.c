#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 256
#define INITIAL_CAPACITY 10

typedef struct {
    char **lines;
    int count;
    int capacity;
} LineBuffer;

typedef struct {
    int total_lines;
    int total_chars;
    int total_words;
    double avg_line_length;
} Statistics;

LineBuffer* create_line_buffer() {
    LineBuffer *buffer = (LineBuffer*)malloc(sizeof(LineBuffer));
    if (!buffer) {
        return NULL;
    }
    buffer->lines = (char**)malloc(INITIAL_CAPACITY * sizeof(char*));
    if (!buffer->lines) {
        free(buffer);
        return NULL;
    }
    buffer->count = 0;
    buffer->capacity = INITIAL_CAPACITY;
    return buffer;
}

int add_line(LineBuffer *buffer, const char *line) {
    if (buffer->count >= buffer->capacity) {
        buffer->capacity *= 2;
        char **new_lines = (char**)realloc(buffer->lines, buffer->capacity * sizeof(char*));
        if (!new_lines) {
            return 0;
        }
        buffer->lines = new_lines;
    }
    
    buffer->lines[buffer->count] = (char*)malloc(strlen(line) + 1);
    if (!buffer->lines[buffer->count]) {
        return 0;
    }
    strcpy(buffer->lines[buffer->count], line);
    buffer->count++;
    return 1;
}

int count_words(const char *line) {
    int count = 0;
    int in_word = 0;
    
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == ' ' || line[i] == '\t' || line[i] == '\n') {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            count++;
        }
    }
    return count;
}

Statistics* calculate_statistics(LineBuffer *buffer) {
    Statistics *stats = (Statistics*)malloc(sizeof(Statistics));
    if (!stats) {
        return NULL;
    }
    
    stats->total_lines = buffer->count;
    stats->total_chars = 0;
    stats->total_words = 0;
    
    for (int i = 0; i < buffer->count; i++) {
        int line_len = strlen(buffer->lines[i]);
        stats->total_chars += line_len;
        stats->total_words += count_words(buffer->lines[i]);
    }
    
    if (stats->total_lines > 0) {
        stats->avg_line_length = (double)stats->total_chars / stats->total_lines;
    } else {
        stats->avg_line_length = 0.0;
    }
    
    return stats;
}

void print_statistics(Statistics *stats) {
    printf("=== File Processing Statistics ===\n");
    printf("Total lines: %d\n", stats->total_lines);
    printf("Total characters: %d\n", stats->total_chars);
    printf("Total words: %d\n", stats->total_words);
    printf("Average line length: %.2f\n", stats->avg_line_length);
    printf("==================================\n");
}

void free_line_buffer(LineBuffer *buffer) {
    if (buffer) {
        for (int i = 0; i < buffer->count; i++) {
            free(buffer->lines[i]);
        }
        free(buffer->lines);
        free(buffer);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file '%s'\n", argv[1]);
        return 1;
    }
    
    LineBuffer *buffer = create_line_buffer();
    if (!buffer) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return 1;
    }
    
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        if (!add_line(buffer, line)) {
            fprintf(stderr, "Error: Failed to add line to buffer\n");
            free_line_buffer(buffer);
            fclose(file);
            return 1;
        }
    }
    
    fclose(file);
    
    Statistics *stats = calculate_statistics(buffer);
    if (!stats) {
        fprintf(stderr, "Error: Failed to calculate statistics\n");
        free_line_buffer(buffer);
        return 1;
    }
    
    print_statistics(stats);
    
    free_line_buffer(buffer);
    // Memory leak: forgot to free stats
    
    return 0;
}