#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

/* String utility functions */
int string_length(char *str);
void string_copy(char *dest, char *src);

/* Array utility functions */
int calculate_sum(int *array, int size);
int find_maximum(int a, int b, int c);

/* Additional utility function */
void print_array(int *array, int size);

#endif /* UTILS_H */