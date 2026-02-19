#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "processor.h"

int process_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        return -1;
    }

    long total_chars = 0;
    long total_lines = 0;
    long total_words = 0;
    long total_letters = 0;
    int in_word = 0;
    int c;
    int prev_char = '\n';

    while ((c = fgetc(file)) != EOF) {
        total_chars++;

        if (c == '\n') {
            total_lines++;
        }

        if (isalpha(c)) {
            total_letters++;
        }

        if (isspace(c)) {
            if (in_word) {
                total_words++;
                in_word = 0;
            }
        } else {
            in_word = 1;
        }

        prev_char = c;
    }

    if (in_word) {
        total_words++;
    }

    if (prev_char != '\n' && total_chars > 0) {
        total_lines++;
    }

    fclose(file);

    printf("=== File Statistics for '%s' ===\n", filename);
    printf("Total characters: %ld\n", total_chars);
    printf("Total lines: %ld\n", total_lines);
    printf("Total words: %ld\n", total_words);
    printf("Total letters: %ld\n", total_letters);

    if (total_words > 0) {
        double avg_word_length = (double)total_letters / (double)total_words;
        printf("Average word length: %.2f letters\n", avg_word_length);
    } else {
        printf("Average word length: 0.00 letters\n");
    }

    if (total_chars > 0) {
        double letter_percentage = ((double)total_letters / (double)total_chars) * 100.0;
        printf("Letter percentage: %.2f%%\n", letter_percentage);
    } else {
        printf("Letter percentage: 0.00%%\n");
    }

    printf("================================\n");

    return 0;
}