#include "utils.h"
#include <string.h>
#include <stdlib.h>

/* Used static data arrays */
static const int prime_numbers[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
static const int prime_count = 15;

static const double conversion_factors[] = {1.0, 2.54, 0.3937, 100.0, 0.01};
static const int factor_count = 5;

/* Unused static data arrays */
static const char* legacy_error_messages[] = {
    "Legacy error 1",
    "Legacy error 2",
    "Legacy error 3",
    "Deprecated message",
    "Old style warning"
};

static const int unused_lookup_table[] = {
    10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    110, 120, 130, 140, 150, 160, 170, 180, 190, 200
};

static const double unused_coefficients[] = {
    0.123, 0.456, 0.789, 1.234, 5.678, 9.012
};

/* ========== USED FUNCTIONS (called from main.c) ========== */

int string_length(const char* str) {
    if (!str) return 0;
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

int array_sum(const int* arr, int size) {
    if (!arr || size <= 0) return 0;
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

int find_max(const int* arr, int size) {
    if (!arr || size <= 0) return 0;
    int max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

double convert_value(double value, int factor_index) {
    if (factor_index < 0 || factor_index >= factor_count) {
        return value;
    }
    return value * conversion_factors[factor_index];
}

int is_prime_number(int num) {
    for (int i = 0; i < prime_count; i++) {
        if (prime_numbers[i] == num) {
            return 1;
        }
    }
    return 0;
}

/* ========== UNUSED FUNCTIONS (never called) ========== */

char* old_parser(const char* input, int mode) {
    if (!input) return NULL;
    int len = string_length(input);
    char* result = (char*)malloc(len + 1);
    if (!result) return NULL;
    
    for (int i = 0; i < len; i++) {
        if (mode == 1) {
            result[i] = input[i] + 1;
        } else if (mode == 2) {
            result[i] = input[i] - 1;
        } else {
            result[i] = input[i];
        }
    }
    result[len] = '\0';
    return result;
}

int legacy_converter(int value, int base) {
    int result = 0;
    int multiplier = 1;
    
    while (value > 0) {
        int digit = value % 10;
        result += digit * multiplier;
        multiplier *= base;
        value /= 10;
    }
    
    return result;
}

double unused_helper(double x, double y, double z) {
    double result = 0.0;
    for (int i = 0; i < 6; i++) {
        result += unused_coefficients[i] * (x + y + z);
    }
    return result / 6.0;
}

void deprecated_utility(int* buffer, int size) {
    if (!buffer || size <= 0) return;
    
    for (int i = 0; i < size; i++) {
        buffer[i] = unused_lookup_table[i % 20];
    }
}

const char* get_legacy_error(int error_code) {
    if (error_code < 0 || error_code >= 5) {
        return "Unknown error";
    }
    return legacy_error_messages[error_code];
}

int obsolete_calculator(int a, int b, char operation) {
    switch (operation) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return (b != 0) ? a / b : 0;
        case '%': return (b != 0) ? a % b : 0;
        default: return 0;
    }
}

void old_style_sort(int* arr, int size) {
    if (!arr || size <= 1) return;
    
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

double deprecated_math_function(double input) {
    double result = input;
    for (int i = 0; i < 10; i++) {
        result = result * 1.1 - 0.05;
    }
    return result;
}