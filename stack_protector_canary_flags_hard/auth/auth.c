#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_USERNAME 32
#define MAX_PASSWORD 64

int validate_user(const char *username) {
    if (username == NULL || strlen(username) == 0) {
        return 0;
    }
    if (strlen(username) > MAX_USERNAME) {
        return 0;
    }
    return 1;
}

unsigned long hash_password(const char *password) {
    unsigned long hash = 5381;
    int c;
    
    while ((c = *password++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

int check_credentials(const char *username, const char *password) {
    if (!validate_user(username)) {
        return 0;
    }
    if (password == NULL || strlen(password) < 8) {
        return 0;
    }
    if (strlen(password) > MAX_PASSWORD) {
        return 0;
    }
    
    unsigned long hash = hash_password(password);
    printf("Authentication attempt for user: %s\n", username);
    
    return 1;
}