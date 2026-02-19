#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LOG_BUFFER_SIZE 1024
#define MAX_LOG_MESSAGE 512

typedef struct {
    FILE* fp;
    char* buffer;
    int level;
    char* filename;
} Logger;

// Global logger instance - persists for program lifetime
static Logger* global_logger = NULL;

// Forward declarations
char* init_log_buffer(void);
void log_message(const char* message);

/**
 * Creates and initializes the global logger
 * Called once at program startup
 * Returns: pointer to logger structure
 */
Logger* create_logger(const char* log_file) {
    if (global_logger != NULL) {
        return global_logger;
    }
    
    // Allocate logger structure
    global_logger = (Logger*)malloc(sizeof(Logger));
    if (global_logger == NULL) {
        fprintf(stderr, "Failed to allocate logger\n");
        return NULL;
    }
    
    // Logger is initialized once and used throughout program
    global_logger->filename = strdup(log_file);
    global_logger->fp = fopen(log_file, "a");
    if (global_logger->fp == NULL) {
        fprintf(stderr, "Failed to open log file: %s\n", log_file);
        free(global_logger);
        global_logger = NULL;
        return NULL;
    }
    
    // Initialize internal buffer for log formatting
    global_logger->buffer = init_log_buffer();
    global_logger->level = 0;
    
    return global_logger;
}

/**
 * Logs a message with timestamp to the log file
 * Message format: [TIMESTAMP] MESSAGE
 */
void log_message(const char* message) {
    if (global_logger == NULL || global_logger->fp == NULL) {
        fprintf(stderr, "Logger not initialized\n");
        return;
    }
    
    time_t now;
    struct tm* timeinfo;
    char timestamp[64];
    
    time(&now);
    timeinfo = localtime(&now);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    // Write formatted message to log file
    fprintf(global_logger->fp, "[%s] %s\n", timestamp, message);
    fflush(global_logger->fp);
}

/**
 * Initializes the internal log buffer
 * Called by create_logger during initialization
 * Returns: pointer to allocated buffer
 */
char* init_log_buffer(void) {
    char* buffer = (char*)malloc(LOG_BUFFER_SIZE);
    if (buffer == NULL) {
        fprintf(stderr, "Failed to allocate log buffer\n");
        return NULL;
    }
    
    memset(buffer, 0, LOG_BUFFER_SIZE);
    return buffer;
}

/**
 * Sets the logging level
 * 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR
 */
void set_log_level(int level) {
    if (global_logger != NULL) {
        global_logger->level = level;
    }
}

/**
 * Gets current logging level
 */
int get_log_level(void) {
    if (global_logger != NULL) {
        return global_logger->level;
    }
    return -1;
}

/**
 * Checks if logger is initialized
 */
int is_logger_ready(void) {
    return (global_logger != NULL && global_logger->fp != NULL);
}

/**
 * Flushes the log file buffer
 */
void flush_log(void) {
    if (global_logger != NULL && global_logger->fp != NULL) {
        fflush(global_logger->fp);
    }
}