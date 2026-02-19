#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 256
#define MAX_PATTERN 64

typedef struct {
    int total_lines;
    int error_count;
    int warning_count;
    int info_count;
} LogStats;

void print_banner() {
    printf("=================================\n");
    printf("  Server Log Parser v1.2\n");
    printf("=================================\n\n");
}

void process_line(char *line, LogStats *stats) {
    char temp_buffer[MAX_LINE];
    char pattern[MAX_PATTERN];
    
    stats->total_lines++;
    
    // Bug 1: strcpy without bounds checking - can overflow temp_buffer
    strcpy(temp_buffer, line);
    
    // Count ERROR patterns
    if (strstr(temp_buffer, "ERROR") != NULL) {
        stats->error_count++;
        // Bug 2: sprintf without size limit - can overflow pattern buffer
        sprintf(pattern, "Found ERROR in line: %s", temp_buffer);
    }
    
    // Count WARNING patterns
    if (strstr(temp_buffer, "WARNING") != NULL) {
        stats->warning_count++;
    }
    
    // Count INFO patterns
    if (strstr(temp_buffer, "INFO") != NULL) {
        stats->info_count++;
    }
    
    // Extract and process IP addresses (if present)
    char *ip_start = strstr(temp_buffer, "IP:");
    if (ip_start != NULL) {
        char ip_buffer[32];
        int i = 0;
        ip_start += 3; // Skip "IP:"
        
        // Bug 3: Off-by-one error - can write beyond buffer
        while (ip_start[i] != ' ' && ip_start[i] != '\n' && ip_start[i] != '\0') {
            ip_buffer[i] = ip_start[i];
            i++;
        }
        ip_buffer[i] = '\0';
    }
}

void parse_log_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    char line[MAX_LINE];
    LogStats stats = {0, 0, 0, 0};
    
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }
    
    printf("Processing log file: %s\n", filename);
    printf("-----------------------------------\n");
    
    // Bug 4: fgets with wrong buffer size - reading more than allocated
    while (fgets(line, 512, file) != NULL) {
        // Remove newline if present
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }
        
        process_line(line, &stats);
    }
    
    fclose(file);
    
    // Print statistics
    printf("\nLog Statistics:\n");
    printf("-----------------------------------\n");
    printf("Total lines processed: %d\n", stats.total_lines);
    printf("ERROR entries found:   %d\n", stats.error_count);
    printf("WARNING entries found: %d\n", stats.warning_count);
    printf("INFO entries found:    %d\n", stats.info_count);
    printf("===================================\n\n");
}

void analyze_pattern_frequency(const char *filename) {
    FILE *file = fopen(filename, "r");
    char line[MAX_LINE];
    char all_lines[2048];
    
    if (file == NULL) {
        return;
    }
    
    all_lines[0] = '\0';
    
    // Bug 5: strcat without checking total buffer size - can overflow all_lines
    while (fgets(line, MAX_LINE, file) != NULL) {
        strcat(all_lines, line);
    }
    
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <log_file>\n", argv[0]);
        fprintf(stderr, "Example: %s /var/log/server.log\n", argv[0]);
        return 1;
    }
    
    print_banner();
    
    parse_log_file(argv[1]);
    
    // Additional analysis
    analyze_pattern_frequency(argv[1]);
    
    printf("Analysis complete.\n");
    
    return 0;
}