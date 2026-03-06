#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "utils.h"

int string_length(const char *str) {
    int count = 0;
    int unused_var = 42;
    
    if (str == NULL) {
        return 0;
    }
    
    while (*str != '\0') {
        count++;
        str++;
    }
    
    return count;
}

int validate_positive_number(int num) {
    if (num > 0) {
        return 1;
    }
    return 0;
}

char* string_to_upper(char *str) {
    int i;
    
    if (str == NULL) {
        return NULL;
    }
    
    for (i = 0; i < strlen(str); i++) {
        str[i] = toupper((unsigned char)str[i]);
    }
    
    return str;
}

int count_digits(const char *str) {
    int count = 0;
    int length = strlen(str);
    int i;
    
    if (str == NULL) {
        return 0;
    }
    
    for (i = 0; i < length; i++) {
        if (isdigit((unsigned char)str[i])) {
            count++;
        }
    }
    
    return count;
}

int compare_numbers(int a, int b, int precision) {
    if (a > b) {
        return 1;
    } else if (a < b) {
        return -1;
    }
    return 0;
}

void print_array(int *arr, unsigned int size, int verbose) {
    int i;
    
    if (arr == NULL) {
        return;
    }
    
    printf("Array contents: ");
    for (i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int safe_string_copy(char *dest, const char *src, int max_len) {
    int copied = 0;
    
    if (dest == NULL || src == NULL || max_len <= 0) {
        return -1;
    }
    
    while (*src != '\0' && copied < max_len - 1) {
        *dest++ = *src++;
        copied++;
    }
    
    *dest = '\0';
    return copied;
}