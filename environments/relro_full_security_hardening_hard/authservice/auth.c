#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_USERNAME_LENGTH 64
#define MAX_PASSWORD_LENGTH 128
#define MAX_USERS 100
#define MIN_PASSWORD_LENGTH 8

typedef struct {
    char username[MAX_USERNAME_LENGTH];
    char password_hash[MAX_PASSWORD_LENGTH];
    int active;
} User;

typedef struct {
    User users[MAX_USERS];
    int user_count;
    int initialized;
} AuthService;

static AuthService auth_service = {0};

// Simple hash function for demonstration (not cryptographically secure)
unsigned long hash_password(const char* password) {
    unsigned long hash = 5381;
    int c;
    
    if (password == NULL) {
        return 0;
    }
    
    while ((c = *password++)) {
        hash = ((hash << 5) + hash) + c;
    }
    
    return hash;
}

int init_auth_service(void) {
    printf("[AUTH SERVICE] Initializing authentication service...\n");
    
    if (auth_service.initialized) {
        printf("[AUTH SERVICE] Service already initialized\n");
        return 1;
    }
    
    memset(&auth_service, 0, sizeof(AuthService));
    auth_service.user_count = 0;
    auth_service.initialized = 1;
    
    // Add some default test users
    strncpy(auth_service.users[0].username, "admin", MAX_USERNAME_LENGTH - 1);
    snprintf(auth_service.users[0].password_hash, MAX_PASSWORD_LENGTH, "%lu", hash_password("admin123"));
    auth_service.users[0].active = 1;
    
    strncpy(auth_service.users[1].username, "testuser", MAX_USERNAME_LENGTH - 1);
    snprintf(auth_service.users[1].password_hash, MAX_PASSWORD_LENGTH, "%lu", hash_password("testpass123"));
    auth_service.users[1].active = 1;
    
    auth_service.user_count = 2;
    
    printf("[AUTH SERVICE] Service initialized successfully with %d users\n", auth_service.user_count);
    return 0;
}

int validate_input(const char* input, int max_length) {
    if (input == NULL) {
        printf("[AUTH SERVICE] Validation failed: NULL input\n");
        return 0;
    }
    
    int length = strlen(input);
    if (length == 0 || length >= max_length) {
        printf("[AUTH SERVICE] Validation failed: Invalid input length\n");
        return 0;
    }
    
    return 1;
}

int find_user(const char* username) {
    if (!validate_input(username, MAX_USERNAME_LENGTH)) {
        return -1;
    }
    
    for (int i = 0; i < auth_service.user_count; i++) {
        if (auth_service.users[i].active && 
            strncmp(auth_service.users[i].username, username, MAX_USERNAME_LENGTH) == 0) {
            return i;
        }
    }
    
    return -1;
}

int validate_credentials(const char* username, const char* password) {
    printf("[AUTH SERVICE] Validating credentials for user: %s\n", username);
    
    if (!auth_service.initialized) {
        printf("[AUTH SERVICE] Error: Service not initialized\n");
        return -1;
    }
    
    if (!validate_input(username, MAX_USERNAME_LENGTH)) {
        printf("[AUTH SERVICE] Invalid username format\n");
        return 0;
    }
    
    if (!validate_input(password, MAX_PASSWORD_LENGTH)) {
        printf("[AUTH SERVICE] Invalid password format\n");
        return 0;
    }
    
    if (strlen(password) < MIN_PASSWORD_LENGTH) {
        printf("[AUTH SERVICE] Password too short\n");
        return 0;
    }
    
    int user_index = find_user(username);
    if (user_index < 0) {
        printf("[AUTH SERVICE] User not found: %s\n", username);
        return 0;
    }
    
    char password_hash_str[MAX_PASSWORD_LENGTH];
    snprintf(password_hash_str, MAX_PASSWORD_LENGTH, "%lu", hash_password(password));
    
    if (strncmp(auth_service.users[user_index].password_hash, password_hash_str, MAX_PASSWORD_LENGTH) == 0) {
        printf("[AUTH SERVICE] Authentication successful for user: %s\n", username);
        return 1;
    }
    
    printf("[AUTH SERVICE] Authentication failed for user: %s\n", username);
    return 0;
}

int process_auth_request(const char* username, const char* password) {
    time_t current_time;
    time(&current_time);
    
    printf("[AUTH SERVICE] Processing authentication request at %s", ctime(&current_time));
    
    int result = validate_credentials(username, password);
    
    if (result == 1) {
        printf("[AUTH SERVICE] Access granted\n");
        return 0;
    } else {
        printf("[AUTH SERVICE] Access denied\n");
        return 1;
    }
}

void shutdown_auth_service(void) {
    printf("[AUTH SERVICE] Shutting down authentication service...\n");
    
    if (auth_service.initialized) {
        memset(&auth_service, 0, sizeof(AuthService));
        printf("[AUTH SERVICE] Service shut down successfully\n");
    }
}

void print_service_info(void) {
    printf("========================================\n");
    printf("  Authentication Service v1.0\n");
    printf("  Critical Security Component\n");
    printf("  Production Deployment Build\n");
    printf("========================================\n");
}

int main(int argc, char* argv[]) {
    print_service_info();
    
    printf("[AUTH SERVICE] Starting authentication service...\n");
    
    if (init_auth_service() != 0) {
        fprintf(stderr, "[AUTH SERVICE] Failed to initialize service\n");
        return 1;
    }
    
    printf("[AUTH SERVICE] Running authentication tests...\n");
    
    // Test case 1: Valid admin credentials
    printf("\n--- Test 1: Valid admin credentials ---\n");
    process_auth_request("admin", "admin123");
    
    // Test case 2: Valid test user credentials
    printf("\n--- Test 2: Valid test user credentials ---\n");
    process_auth_request("testuser", "testpass123");
    
    // Test case 3: Invalid password
    printf("\n--- Test 3: Invalid password ---\n");
    process_auth_request("admin", "wrongpassword");
    
    // Test case 4: Non-existent user
    printf("\n--- Test 4: Non-existent user ---\n");
    process_auth_request("nonexistent", "somepassword");
    
    // Test case 5: NULL input validation
    printf("\n--- Test 5: Input validation tests ---\n");
    validate_credentials(NULL, "password");
    validate_credentials("username", NULL);
    
    printf("\n[AUTH SERVICE] All tests completed\n");
    
    shutdown_auth_service();
    
    printf("[AUTH SERVICE] Service terminated successfully\n");
    return 0;
}