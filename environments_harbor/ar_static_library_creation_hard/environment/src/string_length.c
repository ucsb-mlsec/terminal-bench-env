#include <stdio.h>
#include <string.h>

int string_len(const char *str) {
    if (str == NULL) {
        return 0;
    }
    
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    
    return length;
}

int string_length(const char *str) {
    if (str == NULL) {
        return 0;
    }
    return strlen(str);
}

void print_string_length(const char *str) {
    if (str != NULL) {
        printf("String length: %d\n", string_len(str));
    }
}