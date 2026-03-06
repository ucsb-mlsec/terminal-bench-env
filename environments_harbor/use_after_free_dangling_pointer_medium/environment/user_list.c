#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct UserNode {
    int user_id;
    char username[50];
    struct UserNode *next;
} UserNode;

UserNode *head = NULL;

void add_user(int id, const char *name) {
    UserNode *new_user = (UserNode *)malloc(sizeof(UserNode));
    if (new_user == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    new_user->user_id = id;
    strncpy(new_user->username, name, sizeof(new_user->username) - 1);
    new_user->username[sizeof(new_user->username) - 1] = '\0';
    new_user->next = head;
    head = new_user;
    
    printf("Added user: ID=%d, Username=%s\n", id, name);
}

void cleanup_users() {
    printf("\nStarting cleanup process...\n");
    UserNode *current = head;
    int count = 0;
    
    while (current != NULL) {
        printf("Freeing user: ID=%d, Username=%s\n", current->user_id, current->username);
        free(current);
        current = current->next;  // BUG: Use-after-free error
        count++;
    }
    
    printf("Cleanup complete. Freed %d user records.\n", count);
    head = NULL;
}

void print_users() {
    printf("\nCurrent user list:\n");
    UserNode *current = head;
    int count = 0;
    
    while (current != NULL) {
        printf("  User %d: ID=%d, Username=%s\n", ++count, current->user_id, current->username);
        current = current->next;
    }
    printf("Total users: %d\n", count);
}

int main() {
    printf("User Management System - Starting...\n\n");
    
    // Create and add users
    add_user(1001, "alice_smith");
    add_user(1002, "bob_jones");
    add_user(1003, "carol_williams");
    add_user(1004, "david_brown");
    add_user(1005, "eve_davis");
    add_user(1006, "frank_miller");
    add_user(1007, "grace_wilson");
    
    // Display all users
    print_users();
    
    // Cleanup all users
    cleanup_users();
    
    printf("\nProgram completed successfully.\n");
    return 0;
}