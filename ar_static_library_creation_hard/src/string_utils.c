// String library - process function
#include <stdio.h>
#include <string.h>

int process(char *str) {
    if (str == NULL) {
        return 0;
    }
    int length = strlen(str);
    printf("String processing: length = %d\n", length);
    return length;
}

int string_concat(char *dest, const char *src, int max_len) {
    if (dest == NULL || src == NULL) {
        return -1;
    }
    int dest_len = strlen(dest);
    int src_len = strlen(src);
    int copy_len = (dest_len + src_len < max_len) ? src_len : (max_len - dest_len - 1);
    
    if (copy_len > 0) {
        strncat(dest, src, copy_len);
    }
    return strlen(dest);
}

int string_reverse(char *str) {
    if (str == NULL) {
        return -1;
    }
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = temp;
    }
    return len;
}