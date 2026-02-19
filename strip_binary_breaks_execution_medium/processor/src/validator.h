#ifndef VALIDATOR_H
#define VALIDATOR_H

#define MAX_LINE_LENGTH 1024
#define MAX_FIELDS 10

/* Function to validate a single CSV row */
int validate_row(char* row);

#endif