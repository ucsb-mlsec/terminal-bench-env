#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

#define MAX_PASSWORD_LENGTH 100

int check_length(const char *password) {
    int len = strlen(password);
    return (len >= 10 && len <= 15);
}

int check_starts_with(const char *password) {
    return (strncmp(password, "Sec", 3) == 0);
}

int check_ends_with(const char *password) {
    int len = strlen(password);
    if (len < 3) return 0;
    return (strcmp(password + len - 3, "42!") == 0);
}

int check_character_at_position(const char *password) {
    int len = strlen(password);
    if (len < 6) return 0;
    return (password[5] == 'r');
}

int check_ascii_sum(const char *password) {
    int len = strlen(password);
    if (len < 9) return 0;
    
    int sum = 0;
    for (int i = 6; i < 9; i++) {
        sum += (int)password[i];
    }
    
    return (sum == 291);
}

int check_has_uppercase(const char *password) {
    for (int i = 0; password[i] != '\0'; i++) {
        if (isupper(password[i])) return 1;
    }
    return 0;
}

int check_has_digit(const char *password) {
    for (int i = 0; password[i] != '\0'; i++) {
        if (isdigit(password[i])) return 1;
    }
    return 0;
}

int check_special_char(const char *password) {
    for (int i = 0; password[i] != '\0'; i++) {
        if (password[i] == '!' || password[i] == '@' || 
            password[i] == '#' || password[i] == '$' ||
            password[i] == '%' || password[i] == '&' ||
            password[i] == '*' || password[i] == '!') {
            return 1;
        }
    }
    return 0;
}

int validate_password(const char *password) {
    if (!check_length(password)) {
        return 0;
    }
    
    if (!check_starts_with(password)) {
        return 0;
    }
    
    if (!check_ends_with(password)) {
        return 0;
    }
    
    if (!check_character_at_position(password)) {
        return 0;
    }
    
    if (!check_ascii_sum(password)) {
        return 0;
    }
    
    if (!check_has_uppercase(password)) {
        return 0;
    }
    
    if (!check_has_digit(password)) {
        return 0;
    }
    
    if (!check_special_char(password)) {
        return 0;
    }
    
    return 1;
}

int main() {
    char password[MAX_PASSWORD_LENGTH];
    
    if (fgets(password, MAX_PASSWORD_LENGTH, stdin) == NULL) {
        printf("ACCESS DENIED\n");
        return 1;
    }
    
    size_t len = strlen(password);
    if (len > 0 && password[len - 1] == '\n') {
        password[len - 1] = '\0';
    }
    
    if (validate_password(password)) {
        printf("ACCESS GRANTED\n");
        return 0;
    } else {
        printf("ACCESS DENIED\n");
        return 1;
    }
}