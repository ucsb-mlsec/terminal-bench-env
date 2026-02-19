#ifndef STRING_UTILS_H
#define STRING_UTILS_H

/* Reverses a null-terminated string in place */
void str_reverse(char* str);

/* Counts the number of occurrences of a character in a string */
int str_count_char(const char* str, char ch);

/* Converts all lowercase letters in a string to uppercase in place */
void str_toupper(char* str);

#endif