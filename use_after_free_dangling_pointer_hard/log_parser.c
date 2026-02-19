#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "log_parser.h"
#include "stats.h"

LogEntry* parse_log_line(const char* line) {
    if (line == NULL || strlen(line) == 0) {
        return NULL;
    }

    LogEntry* entry = (LogEntry*)malloc(sizeof(LogEntry));
    if (entry == NULL) {
        fprintf(stderr, "Memory allocation failed for LogEntry\n");
        return NULL;
    }

    // Initialize fields
    memset(entry->timestamp, 0, sizeof(entry->timestamp));
    memset(entry->log_level, 0, sizeof(entry->log_level));
    entry->message = NULL;

    // Parse timestamp (first 19 characters: YYYY-MM-DD HH:MM:SS)
    if (strlen(line) < 19) {
        free(entry);
        return NULL;
    }
    strncpy(entry->timestamp, line, 19);
    entry->timestamp[19] = '\0';

    // Find log level (after timestamp)
    const char* level_start = line + 20;
    const char* level_end = strchr(level_start, ' ');
    if (level_end == NULL) {
        free(entry);
        return NULL;
    }

    size_t level_len = level_end - level_start;
    if (level_len >= sizeof(entry->log_level)) {
        level_len = sizeof(entry->log_level) - 1;
    }
    strncpy(entry->log_level, level_start, level_len);
    entry->log_level[level_len] = '\0';

    // Parse message (rest of the line)
    const char* msg_start = level_end + 1;
    size_t msg_len = strlen(msg_start);
    
    // Remove trailing newline if present
    while (msg_len > 0 && (msg_start[msg_len-1] == '\n' || msg_start[msg_len-1] == '\r')) {
        msg_len--;
    }

    entry->message = (char*)malloc(msg_len + 1);
    if (entry->message == NULL) {
        free(entry);
        return NULL;
    }
    strncpy(entry->message, msg_start, msg_len);
    entry->message[msg_len] = '\0';

    return entry;
}

void free_log_entry(LogEntry* entry) {
    if (entry != NULL) {
        if (entry->message != NULL) {
            free(entry->message);
        }
        free(entry);
    }
}

int process_log_file(const char* filename, LogStats* stats) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return -1;
    }

    char buffer[256];
    int line_count = 0;
    int error_count = 0;

    printf("Processing log file: %s\n", filename);

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        line_count++;

        // Skip empty lines
        if (strlen(buffer) <= 1) {
            continue;
        }

        LogEntry* entry = parse_log_line(buffer);
        if (entry == NULL) {
            fprintf(stderr, "Failed to parse line %d\n", line_count);
            error_count++;
            continue;
        }

        // Update statistics with this entry
        update_stats(stats, entry);

        // Process the entry for any special patterns
        if (strcmp(entry->log_level, "ERROR") == 0) {
            // Log error details
            printf("Error found at line %d: %s\n", line_count, entry->message);
        }

        // Check for warning patterns
        if (strcmp(entry->log_level, "WARN") == 0) {
            // Check if this is a retry warning
            if (strstr(entry->message, "Retry") != NULL) {
                printf("Retry warning detected at line %d\n", line_count);
            }
        }

        // Free the entry after processing
        free_log_entry(entry);

        // BUG: Use-after-free - accessing entry after it has been freed
        // This causes a segmentation fault when processing certain log patterns
        if (strcmp(entry->log_level, "ERROR") == 0 || strcmp(entry->log_level, "WARN") == 0) {
            // Still trying to access the freed memory
            printf("Critical entry processed: %s - %s\n", entry->log_level, entry->message);
        }

        // Additional processing that might also access freed memory
        if (entry->message != NULL && strlen(entry->message) > 50) {
            printf("Long message detected\n");
        }
    }

    fclose(file);

    printf("Processed %d lines with %d errors\n", line_count, error_count);
    return 0;
}

void print_statistics(const LogStats* stats) {
    printf("\n=== Log Analysis Statistics ===\n");
    printf("Total entries: %d\n", stats->total_entries);
    printf("INFO entries: %d\n", stats->info_count);
    printf("WARN entries: %d\n", stats->warn_count);
    printf("ERROR entries: %d\n", stats->error_count);
    printf("Unique users: %d\n", stats->unique_users);
    printf("Connection timeouts: %d\n", stats->timeout_count);
    printf("Retry attempts: %d\n", stats->retry_count);
    printf("================================\n\n");
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <log_file>\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];

    // Initialize statistics
    LogStats stats;
    init_stats(&stats);

    // Process the log file
    int result = process_log_file(filename, &stats);
    if (result != 0) {
        fprintf(stderr, "Failed to process log file\n");
        return 1;
    }

    // Print statistics
    print_statistics(&stats);

    // Generate output file with statistics
    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "%s.stats", filename);
    
    FILE* output = fopen(output_filename, "w");
    if (output != NULL) {
        fprintf(output, "Total: %d\n", stats.total_entries);
        fprintf(output, "INFO: %d\n", stats.info_count);
        fprintf(output, "WARN: %d\n", stats.warn_count);
        fprintf(output, "ERROR: %d\n", stats.error_count);
        fprintf(output, "Users: %d\n", stats.unique_users);
        fprintf(output, "Timeouts: %d\n", stats.timeout_count);
        fprintf(output, "Retries: %d\n", stats.retry_count);
        fclose(output);
        printf("Statistics written to: %s\n", output_filename);
    }

    return 0;
}