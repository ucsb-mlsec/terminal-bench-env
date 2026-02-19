#include "../include/output.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Write output to stdout */
static int write_to_stdout(const char *text) {
    if (text == NULL) {
        fprintf(stderr, "Error: NULL text pointer\n");
        return -1;
    }
    
    if (fputs(text, stdout) == EOF) {
        fprintf(stderr, "Error: Failed to write to stdout\n");
        return -1;
    }
    
    return 0;
}

/* Write output to file */
static int write_to_file(const char *text, const char *filename) {
    FILE *fp;
    
    if (text == NULL || filename == NULL) {
        fprintf(stderr, "Error: NULL pointer provided\n");
        return -1;
    }
    
    fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open file '%s' for writing\n", filename);
        return -1;
    }
    
    if (fputs(text, fp) == EOF) {
        fprintf(stderr, "Error: Failed to write to file '%s'\n", filename);
        fclose(fp);
        return -1;
    }
    
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file '%s'\n", filename);
        return -1;
    }
    
    return 0;
}

/* Main output function */
int write_output(const char *formatted_text, const char *output_file) {
    if (formatted_text == NULL) {
        fprintf(stderr, "Error: No text to write\n");
        return -1;
    }
    
    if (output_file == NULL) {
        return write_to_stdout(formatted_text);
    } else {
        return write_to_file(formatted_text, output_file);
    }
}