#ifndef LOG_PARSER_H
#define LOG_PARSER_H

typedef struct {
    char timestamp[20];
    char log_level[10];
    char* message;
} LogEntry;

typedef struct {
    int total_count;
    int info_count;
    int warn_count;
    int error_count;
} LogStats;

LogEntry* parse_log_line(char* line);
void free_log_entry(LogEntry* entry);
void process_log_file(char* filename, LogStats* stats);
void print_stats(LogStats* stats);

#endif