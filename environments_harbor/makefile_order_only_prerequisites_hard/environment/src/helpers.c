#include <string.h>

/**
 * Calculate the length of a string
 * @param str Input string
 * @return Length of the string
 */
int string_length(const char* str) {
    if (str == NULL) {
        return 0;
    }
    return strlen(str);
}

/**
 * Copy a string from source to destination
 * @param dest Destination buffer
 * @param src Source string
 */
void copy_string(char* dest, char* src) {
    if (dest != NULL && src != NULL) {
        strcpy(dest, src);
    }
}