#include <stdio.h>
#include <string.h>
#include "log_parser.h"
#include "stats.h"

void reset_stats(LogStats *stats) {
    if (stats == NULL) {
        return;
    }
    stats->total_count = 0;
    stats->info_count = 0;
    stats->warn_count = 0;
    stats->error_count = 0;
}

void update_stats(LogStats *stats, LogEntry *entry) {
    if (stats == NULL || entry == NULL) {
        return;
    }
    
    stats->total_count++;
    
    if (strcmp(entry->log_level, "INFO") == 0) {
        stats->info_count++;
    } else if (strcmp(entry->log_level, "WARN") == 0) {
        stats->warn_count++;
    } else if (strcmp(entry->log_level, "ERROR") == 0) {
        stats->error_count++;
    }
}

void print_stats(LogStats *stats) {
    if (stats == NULL) {
        return;
    }
    
    printf("Total entries: %d\n", stats->total_count);
    printf("INFO: %d\n", stats->info_count);
    printf("WARN: %d\n", stats->warn_count);
    printf("ERROR: %d\n", stats->error_count);
}