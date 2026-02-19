#include <stdio.h>
#include <string.h>
#include "utils.h"

int utility_func(int value) {
    printf("Utility function called with value: %d\n", value);
    return value * 2;
}

int string_length(const char* str) {
    if (str == NULL) {
        return 0;
    }
    return strlen(str);
}

void print_banner(void) {
    printf("==========================================\n");
    printf("  Project Utility Library v1.0\n");
    printf("==========================================\n");
}