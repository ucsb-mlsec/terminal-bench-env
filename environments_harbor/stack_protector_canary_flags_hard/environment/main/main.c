#include <stdio.h>
#include <stdlib.h>

/* Forward declarations for functions from other components */
extern int validate_user(const char *username, const char *password);
extern int create_socket(int port);
extern int save_data(const char *key, const char *value);

int main(int argc, char *argv[]) {
    printf("Starting application...\n");
    
    /* Authentication component usage */
    printf("Attempting authentication...\n");
    int auth_result = validate_user("admin", "password123");
    if (auth_result != 0) {
        fprintf(stderr, "Authentication failed\n");
        return 1;
    }
    printf("Authentication successful\n");
    
    /* Network component usage */
    printf("Initializing network on port 8080...\n");
    int socket_fd = create_socket(8080);
    if (socket_fd < 0) {
        fprintf(stderr, "Failed to create socket\n");
        return 1;
    }
    printf("Network initialized successfully (fd: %d)\n", socket_fd);
    
    /* Storage component usage */
    printf("Saving configuration data...\n");
    int save_result = save_data("config", "enabled");
    if (save_result != 0) {
        fprintf(stderr, "Failed to save data\n");
        return 1;
    }
    printf("Data saved successfully\n");
    
    printf("Application running normally\n");
    printf("All components initialized and operational\n");
    
    return 0;
}