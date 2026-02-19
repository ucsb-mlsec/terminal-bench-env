#include <stdio.h>
#include <string.h>

void string_concat(const char *str1, const char *str2, char *dest) {
    if (str1 == NULL || str2 == NULL || dest == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to string_concat\n");
        return;
    }
    
    strcpy(dest, str1);
    strcat(dest, str2);
}

int string_length(const char *str) {
    if (str == NULL) {
        return 0;
    }
    return strlen(str);
}

void string_copy(const char *src, char *dest) {
    if (src == NULL || dest == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to string_copy\n");
        return;
    }
    strcpy(dest, src);
}