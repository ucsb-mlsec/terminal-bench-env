#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

#include <stddef.h>

/* Returns the maximum value in the array */
int array_max(const int* arr, size_t len);

/* Returns the sum of all elements in the array */
long array_sum(const int* arr, size_t len);

/* Reverses the elements in the array in-place */
void array_reverse(int* arr, size_t len);

#endif