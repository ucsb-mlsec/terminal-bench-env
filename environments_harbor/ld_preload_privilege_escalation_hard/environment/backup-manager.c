/*
 * Backup Manager Utility
 * Performs automated system backups with elevated privileges
 * WARNING: This utility runs with SUID root permissions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define BACKUP_DIR "/var/backups"
#define LOG_FILE "/var/log/backup-manager.log"
#define MAX_PATH 256

/* Function to log backup operations */
void log_operation(const char *message) {
    FILE *log = fopen(LOG_FILE, "a");
    if (log != NULL) {
        fprintf(log, "[BACKUP] %s\n", message);
        fclose(log);
    }
}

/* Check if backup directory exists */
int check_backup_directory() {
    char command[MAX_PATH];
    snprintf(command, sizeof(command), "test -d %s", BACKUP_DIR);
    
    // Vulnerable: Using system() without absolute path validation
    if (system(command) != 0) {
        printf("Backup directory does not exist. Creating...\n");
        snprintf(command, sizeof(command), "mkdir -p %s", BACKUP_DIR);
        system(command);
    }
    return 0;
}

/* Perform database backup using external tool */
int backup_database() {
    char backup_file[MAX_PATH];
    snprintf(backup_file, sizeof(backup_file), "%s/database_backup.sql", BACKUP_DIR);
    
    printf("Starting database backup...\n");
    log_operation("Database backup initiated");
    
    // Vulnerable: Using system() to call external command
    char command[512];
    snprintf(command, sizeof(command), 
             "mysqldump -u backup_user --all-databases > %s", backup_file);
    system(command);
    
    printf("Database backup completed.\n");
    return 0;
}

/* Compress backup files */
int compress_backups() {
    printf("Compressing backup files...\n");
    log_operation("Compression started");
    
    // Vulnerable: Using system() to call tar
    char command[512];
    snprintf(command, sizeof(command), 
             "tar -czf %s/backup-$(date +%%Y%%m%%d).tar.gz %s/*.sql", 
             BACKUP_DIR, BACKUP_DIR);
    
    int result = system(command);
    
    if (result == 0) {
        printf("Compression successful.\n");
        log_operation("Compression completed successfully");
    } else {
        fprintf(stderr, "Compression failed with code: %d\n", result);
        log_operation("Compression failed");
    }
    
    return result;
}

/* Sync backups to remote server */
int sync_to_remote() {
    printf("Syncing backups to remote server...\n");
    log_operation("Remote sync initiated");
    
    // Vulnerable: Using system() to call rsync
    char command[512];
    snprintf(command, sizeof(command), 
             "rsync -avz %s/ backup-server:/remote/backups/", BACKUP_DIR);
    
    system(command);
    
    printf("Remote sync completed.\n");
    return 0;
}

int main(int argc, char *argv[]) {
    printf("=== Backup Manager v1.0 ===\n");
    printf("Running with UID: %d, EUID: %d\n", getuid(), geteuid());
    
    log_operation("Backup Manager started");
    
    // Check and create backup directory if needed
    if (check_backup_directory() != 0) {
        fprintf(stderr, "Failed to setup backup directory\n");
        return 1;
    }
    
    // Perform backup operations
    if (backup_database() != 0) {
        fprintf(stderr, "Database backup failed\n");
        return 1;
    }
    
    // Compress the backup files
    if (compress_backups() != 0) {
        fprintf(stderr, "Backup compression failed\n");
        return 1;
    }
    
    // Sync to remote server
    if (sync_to_remote() != 0) {
        fprintf(stderr, "Remote sync failed\n");
        return 1;
    }
    
    log_operation("Backup Manager completed successfully");
    printf("=== All backup operations completed ===\n");
    
    return 0;
}