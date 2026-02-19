#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

// External function from libmath
extern int add(int a, int b);

// Returns the length of a string
int str_length(const char *str) {
    if (str == NULL) {
        return 0;
    }
    return strlen(str);
}

// Returns a newly allocated reversed string
char* str_reverse(const char *str) {
    if (str == NULL) {
        return NULL;
    }
    
    int len = strlen(str);
    char *reversed = (char*)malloc((len + 1) * sizeof(char));
    
    if (reversed == NULL) {
        return NULL;
    }
    
    for (int i = 0; i < len; i++) {
        reversed[i] = str[len - 1 - i];
    }
    reversed[len] = '\0';
    
    return reversed;
}

// Returns a newly allocated string repeated 'times' times
// Uses libmath's add() function to demonstrate dependency
char* str_repeat(const char *str, int times) {
    if (str == NULL || times <= 0) {
        return NULL;
    }
    
    int base_len = strlen(str);
    int total_len = 0;
    
    // Use add() from libmath to calculate total length
    for (int i = 0; i < times; i++) {
        total_len = add(total_len, base_len);
    }
    
    char *result = (char*)malloc((total_len + 1) * sizeof(char));
    if (result == NULL) {
        return NULL;
    }
    
    result[0] = '\0';
    for (int i = 0; i < times; i++) {
        strcat(result, str);
    }
    
    return result;
}

// Converts string to uppercase, returns new allocated string
char* str_upper(const char *str) {
    if (str == NULL) {
        return NULL;
    }
    
    int len = strlen(str);
    char *upper = (char*)malloc((len + 1) * sizeof(char));
    
    if (upper == NULL) {
        return NULL;
    }
    
    for (int i = 0; i < len; i++) {
        upper[i] = toupper((unsigned char)str[i]);
    }
    upper[len] = '\0';
    
    return upper;
}