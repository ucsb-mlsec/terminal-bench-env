#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void internal_normalize(double* array, int size) {
    if (array == NULL || size <= 0) return;
    
    double max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }
    
    if (max_val == 0.0) return;
    
    for (int i = 0; i < size; i++) {
        array[i] /= max_val;
    }
}

double internal_clamp(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

void internal_swap(double* a, double* b) {
    if (a == NULL || b == NULL) return;
    double temp = *a;
    *a = *b;
    *b = temp;
}

double helper_find_max(double* array, int size) {
    if (array == NULL || size <= 0) return 0.0;
    
    double max = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max) {
            max = array[i];
        }
    }
    return max;
}

double helper_find_min(double* array, int size) {
    if (array == NULL || size <= 0) return 0.0;
    
    double min = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] < min) {
            min = array[i];
        }
    }
    return min;
}

int utility_compare(double a, double b) {
    const double epsilon = 1e-9;
    if (a - b > epsilon) return 1;
    if (b - a > epsilon) return -1;
    return 0;
}

void* internal_allocate_temp(int size) {
    if (size <= 0) return NULL;
    void* ptr = malloc(size);
    if (ptr != NULL) {
        memset(ptr, 0, size);
    }
    return ptr;
}

void internal_free_temp(void* ptr) {
    if (ptr != NULL) {
        free(ptr);
    }
}