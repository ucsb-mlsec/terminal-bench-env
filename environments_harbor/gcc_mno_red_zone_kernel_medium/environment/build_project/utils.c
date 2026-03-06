#include <stdio.h>
#include <string.h>

int add_numbers(int a, int b) {
    return a + b;
}

void print_message(const char* msg) {
    if (msg != NULL) {
        printf("%s\n", msg);
    }
}

int str_len(const char* s) {
    int len = 0;
    if (s == NULL) {
        return 0;
    }
    while (s[len] != '\0') {
        len++;
    }
    return len;
}

int string_compare(const char* s1, const char* s2) {
    if (s1 == NULL || s2 == NULL) {
        return -1;
    }
    while (*s1 && (*s1 == *s2)) {
        s1++;
        s2++;
    }
    return *(unsigned char*)s1 - *(unsigned char*)s2;
}