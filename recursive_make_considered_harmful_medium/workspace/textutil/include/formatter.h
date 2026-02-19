#ifndef FORMATTER_H
#define FORMATTER_H

/* Formatter module - handles text formatting operations */

/* Format options structure */
typedef struct {
    int line_width;
    int indent_level;
    char separator;
} format_options_t;

/* Format parsed text data with specified options */
char* format_text(const char* parsed_data, const format_options_t* options);

/* Free memory allocated by format_text */
void free_formatted_text(char* formatted);

#endif /* FORMATTER_H */