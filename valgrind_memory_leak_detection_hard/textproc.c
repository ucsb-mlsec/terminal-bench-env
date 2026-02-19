#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 1024
#define MAX_WORDS 10000

typedef struct {
    char *word;
    int count;
} WordIndex;

char* read_file(const char *filename, long *file_size) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open input file %s\n", filename);
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    *file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    char *content = (char*)malloc(*file_size + 1);
    fread(content, 1, *file_size, fp);
    content[*file_size] = '\0';
    
    fclose(fp);
    return content;
}

char** parse_lines(const char *content, int *line_count) {
    *line_count = 0;
    const char *ptr = content;
    
    while (*ptr) {
        if (*ptr == '\n') (*line_count)++;
        ptr++;
    }
    (*line_count)++;
    
    char **lines = (char**)malloc(sizeof(char*) * (*line_count));
    
    int idx = 0;
    const char *line_start = content;
    ptr = content;
    
    while (*ptr) {
        if (*ptr == '\n') {
            int len = ptr - line_start;
            char *line = (char*)malloc(len + 1);
            strncpy(line, line_start, len);
            line[len] = '\0';
            lines[idx++] = line;
            line_start = ptr + 1;
        }
        ptr++;
    }
    
    if (ptr > line_start) {
        int len = ptr - line_start;
        char *line = (char*)malloc(len + 1);
        strncpy(line, line_start, len);
        line[len] = '\0';
        lines[idx++] = line;
    }
    
    return lines;
}

char* transform_text(const char *input) {
    int len = strlen(input);
    char *output = (char*)malloc(len + 1);
    
    for (int i = 0; i < len; i++) {
        output[i] = toupper(input[i]);
    }
    output[len] = '\0';
    
    return output;
}

char* reverse_words(const char *line) {
    int len = strlen(line);
    char *reversed = (char*)malloc(len + 1);
    char *temp = (char*)malloc(len + 1);
    strcpy(temp, line);
    
    char *words[100];
    int word_count = 0;
    
    char *token = strtok(temp, " \t");
    while (token != NULL && word_count < 100) {
        words[word_count++] = token;
        token = strtok(NULL, " \t");
    }
    
    reversed[0] = '\0';
    for (int i = word_count - 1; i >= 0; i--) {
        strcat(reversed, words[i]);
        if (i > 0) strcat(reversed, " ");
    }
    
    return reversed;
}

WordIndex* build_index(char **lines, int line_count, int *index_size) {
    WordIndex *index = (WordIndex*)malloc(sizeof(WordIndex) * MAX_WORDS);
    *index_size = 0;
    
    for (int i = 0; i < line_count; i++) {
        char *line_copy = (char*)malloc(strlen(lines[i]) + 1);
        strcpy(line_copy, lines[i]);
        
        char *token = strtok(line_copy, " \t\n.,;:!?");
        while (token != NULL) {
            int found = 0;
            for (int j = 0; j < *index_size; j++) {
                if (strcmp(index[j].word, token) == 0) {
                    index[j].count++;
                    found = 1;
                    break;
                }
            }
            
            if (!found && *index_size < MAX_WORDS) {
                index[*index_size].word = (char*)malloc(strlen(token) + 1);
                strcpy(index[*index_size].word, token);
                index[*index_size].count = 1;
                (*index_size)++;
            }
            
            token = strtok(NULL, " \t\n.,;:!?");
        }
    }
    
    return index;
}

int write_output(const char *filename, char **lines, int line_count) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open output file %s\n", filename);
        return 0;
    }
    
    fprintf(fp, "=== TRANSFORMED TEXT ===\n\n");
    
    for (int i = 0; i < line_count; i++) {
        char *transformed = transform_text(lines[i]);
        fprintf(fp, "%s\n", transformed);
    }
    
    fprintf(fp, "\n=== REVERSED WORDS ===\n\n");
    
    for (int i = 0; i < line_count; i++) {
        char *reversed = reverse_words(lines[i]);
        fprintf(fp, "%s\n", reversed);
    }
    
    fclose(fp);
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    
    long file_size;
    char *content = read_file(argv[1], &file_size);
    if (!content) {
        return 1;
    }
    
    int line_count;
    char **lines = parse_lines(content, &line_count);
    
    int index_size;
    WordIndex *word_index = build_index(lines, line_count, &index_size);
    
    printf("Processing file: %s\n", argv[1]);
    printf("Lines read: %d\n", line_count);
    printf("Unique words indexed: %d\n", index_size);
    
    if (!write_output(argv[2], lines, line_count)) {
        return 1;
    }
    
    printf("Output written to: %s\n", argv[2]);
    printf("Processing complete.\n");
    
    return 0;
}