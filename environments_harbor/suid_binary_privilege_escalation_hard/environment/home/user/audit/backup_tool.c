#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>

/* 
 * Backup Tool - Legacy SUID utility for user file backups
 * Version: 1.2
 * 
 * This utility allows users to backup their home directory files
 * to a centralized backup location with proper permissions.
 */

#define BACKUP_DIR "/var/backups/users"
#define MAX_PATH 256

int main(int argc, char *argv[]) {
    char source_path[MAX_PATH];
    char backup_command[512];
    char username[64];
    struct stat st;
    
    // Display usage information
    if (argc != 2) {
        printf("Usage: %s <directory_to_backup>\n", argv[0]);
        printf("Example: %s /home/user/documents\n", argv[0]);
        return 1;
    }
    
    // Get the username of the user running this program
    char *user = getenv("USER");
    if (user == NULL) {
        fprintf(stderr, "Error: Could not determine username\n");
        return 1;
    }
    
    strncpy(username, user, sizeof(username) - 1);
    username[sizeof(username) - 1] = '\0';
    
    // Copy and validate the source path
    strncpy(source_path, argv[1], sizeof(source_path) - 1);
    source_path[sizeof(source_path) - 1] = '\0';
    
    // Check if source directory exists
    if (stat(source_path, &st) != 0) {
        fprintf(stderr, "Error: Source path does not exist: %s\n", source_path);
        return 1;
    }
    
    // Ensure backup directory exists
    if (stat(BACKUP_DIR, &st) != 0) {
        fprintf(stderr, "Error: Backup directory not accessible\n");
        return 1;
    }
    
    printf("Backing up %s for user %s...\n", source_path, username);
    
    // Create backup using tar command
    // The backup will be stored in the centralized backup directory
    snprintf(backup_command, sizeof(backup_command),
             "tar -czf %s/%s-backup.tar.gz %s 2>/dev/null",
             BACKUP_DIR, username, source_path);
    
    // Execute the backup command with elevated privileges
    system(backup_command);
    
    printf("Backup completed successfully!\n");
    printf("Backup file: %s/%s-backup.tar.gz\n", BACKUP_DIR, username);
    
    return 0;
}