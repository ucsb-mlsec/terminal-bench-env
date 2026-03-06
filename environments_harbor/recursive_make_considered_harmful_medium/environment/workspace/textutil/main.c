#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "parser.h"
#include "formatter.h"
#include "output.h"

#define MAX_INPUT_SIZE 4096

/*
 * TextUtil - A simple text processing utility
 * 
 * Program flow:
 * 1. Read input text from stdin or use default test string
 * 2. Parse the text into structured data
 * 3. Format the parsed data according to rules
 * 4. Write the formatted output
 */

int main(int argc, char** argv) {
    char input_buffer[MAX_INPUT_SIZE];
    char* input_text = NULL;
    ParsedData* parsed = NULL;
    FormattedData* formatted = NULL;
    int result = EXIT_SUCCESS;

    /* Determine input source */
    if (argc > 1) {
        /* Use command line argument as input */
        input_text = argv[1];
    } else {
        /* Read from stdin or use default test string */
        printf("Enter text to process (or press Ctrl+D for default): ");
        if (fgets(input_buffer, MAX_INPUT_SIZE, stdin) != NULL) {
            input_text = input_buffer;
        } else {
            /* Use default test string */
            input_text = "Hello World! This is a test of the text processing utility.";
            printf("\nUsing default text: %s\n\n", input_text);
        }
    }

    /* Step 1: Parse the input text */
    printf("Step 1: Parsing text...\n");
    parsed = parse_text(input_text);
    if (parsed == NULL) {
        fprintf(stderr, "Error: Failed to parse input text\n");
        result = EXIT_FAILURE;
        goto cleanup;
    }
    printf("Parsed successfully: %d tokens found\n\n", parsed->token_count);

    /* Step 2: Format the parsed data */
    printf("Step 2: Formatting data...\n");
    formatted = format_text(parsed);
    if (formatted == NULL) {
        fprintf(stderr, "Error: Failed to format parsed data\n");
        result = EXIT_FAILURE;
        goto cleanup;
    }
    printf("Formatted successfully: %d lines generated\n\n", formatted->line_count);

    /* Step 3: Write the output */
    printf("Step 3: Writing output...\n");
    if (write_output(formatted) != 0) {
        fprintf(stderr, "Error: Failed to write output\n");
        result = EXIT_FAILURE;
        goto cleanup;
    }
    printf("Output written successfully\n");

cleanup:
    /* Free allocated resources */
    if (formatted != NULL) {
        free_formatted_data(formatted);
    }
    if (parsed != NULL) {
        free_parsed_data(parsed);
    }

    return result;
}