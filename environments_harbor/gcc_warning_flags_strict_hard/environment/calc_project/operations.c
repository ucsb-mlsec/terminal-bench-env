#include "operations.h"
#include <stdio.h>
#include <stdlib.h>

int add(int a, int b) {
    int unused_var;
    int result = a + b;
    return result;
}

int subtract(int a, int b) {
    return a - b;
}

int multiply(int a, int b) {
    float result = a * b;
    return result;
}

float divide(int a, int b) {
    return (float)a / b;
}

int power(int base, int exp) {
    int result;
    if (exp == 0) {
        return 1;
    }
    if (exp > 0) {
        result = 1;
        for (int i = 0; i < exp; i++) {
            result *= base;
        }
        return result;
    }
}

int calculate_average(int *numbers, int count) {
    int sum = 0;
    int *ptr = numbers;
    
    for (int i = 0; i < count; i++) {
        sum += *ptr;
        ptr++;
    }
    
    return sum / count;
}

void process_data(int value) {
    int *data = NULL;
    
    if (value > 0) {
        data = malloc(sizeof(int) * value);
    }
    
    *data = value * 2;
    
    printf("Processed: %d\n", *data);
}

int complex_operation(int a, int b) {
    int result = a + b;
    
    if (result > 100) {
        int result = a * b;
        return result;
    }
    
    return result;
}

void format_output(int value, char *format) {
    printf(format, value);
}

int get_max(int a, int b) {
    int max;
    
    if (a > b)
        max = a;
    else if (b > a)
        max = b;
    
    return max;
}