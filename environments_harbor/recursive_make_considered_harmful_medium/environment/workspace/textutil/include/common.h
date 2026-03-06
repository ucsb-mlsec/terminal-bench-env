#ifndef COMMON_H
#define COMMON_H

/* Common definitions shared across all textutil modules */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Buffer and string handling constants */
#define MAX_BUFFER_SIZE 1024
#define MAX_LINE_LENGTH 256

/* Return codes */
#define SUCCESS 0
#define ERROR -1

/* Utility function prototypes */
void *safe_malloc(size_t size);
char *safe_strdup(const char *str);

#endif /* COMMON_H */