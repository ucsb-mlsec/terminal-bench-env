#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pwd.h>
#include <sys/types.h>
#include <string.h>
#include <errno.h>

#define HOSTNAME_MAX_LEN 256

int main(void) {
    uid_t uid;
    struct passwd *pw;
    char hostname[HOSTNAME_MAX_LEN];
    
    // Get current user ID
    uid = getuid();
    
    // Get user information from password database
    pw = getpwuid(uid);
    if (pw == NULL) {
        fprintf(stderr, "Error: Unable to get user information for UID %d\n", uid);
        fprintf(stderr, "Error details: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    
    // Get hostname
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        fprintf(stderr, "Error: Unable to get hostname\n");
        fprintf(stderr, "Error details: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    
    // Ensure null termination
    hostname[HOSTNAME_MAX_LEN - 1] = '\0';
    
    // Print system information
    printf("=== System Information ===\n");
    printf("User: %s (UID: %d)\n", pw->pw_name, uid);
    printf("Home: %s\n", pw->pw_dir);
    printf("Shell: %s\n", pw->pw_shell);
    printf("Hostname: %s\n", hostname);
    printf("==========================\n");
    
    return EXIT_SUCCESS;
}