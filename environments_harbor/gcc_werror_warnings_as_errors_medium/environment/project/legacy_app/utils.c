#include <stdio.h>
#include <string.h>

// String length function with unused parameter
int string_length(char *str, int max_len) {
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// String copy with buffer issues
void string_copy(char *dest, char *src) {
    int i = 0;
    while (src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\0';
}

// Function with implicit type conversion
int calculate_sum(int arr[], int size) {
    float total = 0.0;
    int i;
    for (i = 0; i < size; i++) {
        total += arr[i];
    }
    return total;
}

// Function with unused parameter and missing return
int find_maximum(int *arr, int size, int unused_flag) {
    int max = arr[0];
    int i;
    for (i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
}

// Function with pointer and integer comparison
int validate_pointer(char *ptr) {
    if (ptr == 0) {
        return 0;
    }
    return 1;
}

// Format string issue
void print_message(char *msg) {
    printf(msg);
}

// Missing prototype - this will be called from main
void helper_function(int value) {
    printf("Helper value: %d\n", value);
}

// Function with implicit conversion from pointer to int
int get_buffer_status(char *buffer) {
    if (buffer) {
        return 1;
    }
    return buffer;
}

// Function with signed/unsigned comparison
int check_bounds(int index, unsigned int max_size) {
    int limit = -1;
    if (index < limit || index >= max_size) {
        return 0;
    }
    return 1;
}

// Function that should return value but has missing return in some paths
int process_value(int val) {
    if (val > 0) {
        return val * 2;
    } else if (val < 0) {
        return val * -1;
    }
}

// Function with unused variable
int count_chars(char *str, char target, int flags) {
    int count = 0;
    int i;
    int unused_var = 42;
    for (i = 0; str[i] != '\0'; i++) {
        if (str[i] == target) {
            count++;
        }
    }
    return count;
}