#include <stdio.h>
#include <stdlib.h>

void vector_add(const float* a, const float* b, float* result, int size) {
    if (a == NULL || b == NULL || result == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to vector_add\n");
        return;
    }
    
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

void vector_multiply(const float* a, const float* b, float* result, int size) {
    if (a == NULL || b == NULL || result == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to vector_multiply\n");
        return;
    }
    
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

float dot_product(const float* a, const float* b, int size) {
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to dot_product\n");
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

void vector_scale(float* vec, float scalar, int size) {
    if (vec == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to vector_scale\n");
        return;
    }
    
    for (int i = 0; i < size; i++) {
        vec[i] *= scalar;
    }
}

float vector_sum(const float* vec, int size) {
    if (vec == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to vector_sum\n");
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += vec[i];
    }
    
    return sum;
}

void vector_subtract(const float* a, const float* b, float* result, int size) {
    if (a == NULL || b == NULL || result == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to vector_subtract\n");
        return;
    }
    
    for (int i = 0; i < size; i++) {
        result[i] = a[i] - b[i];
    }
}