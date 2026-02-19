#ifndef PARSER_H
#define PARSER_H

/*
 * Parser Module
 * Provides text parsing functionality for the textutil program
 */

/* Structure to hold parsed text data */
typedef struct {
    char* content;
    int length;
} ParsedText;

/* Parse input text and return parsed structure */
ParsedText* parse_text(const char* input);

/* Free memory allocated for parsed text */
void free_parsed_text(ParsedText* parsed);

#endif /* PARSER_H */