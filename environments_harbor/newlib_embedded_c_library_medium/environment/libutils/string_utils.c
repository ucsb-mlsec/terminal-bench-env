#include <string.h>
#include <ctype.h>

void str_reverse(char *str) {
    if (str == NULL) {
        return;
    }
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        char temp = str[i];
        str[i] = str[len - 1 - i];
        str[len - 1 - i] = temp;
    }
}

int str_count_char(const char *str, char c) {
    if (str == NULL) {
        return 0;
    }
    int count = 0;
    while (*str) {
        if (*str == c) {
            count++;
        }
        str++;
    }
    return count;
}

void str_toupper(char *str) {
    if (str == NULL) {
        return;
    }
    while (*str) {
        *str = toupper((unsigned char)*str);
        str++;
    }
}