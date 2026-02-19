#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BUFFER_SIZE 1024
#define DEFAULT_CONFIG_SIZE 256

// Global application state
static void *app_cache = NULL;
static int initialized = 0;

// Function prototypes
int init_application(void);
void cleanup_resources(void);
int process_arguments(int argc, char **argv);
char* process_record(const char *input);
char* allocate_buffer(size_t size);
void setup_cache(void);

/**
 * Initialize application state and resources
 */
int init_application(void) {
    printf("Initializing application...\n");
    
    // Allocate global cache - intentionally kept for program lifetime
    app_cache = malloc(512);
    if (!app_cache) {
        fprintf(stderr, "Failed to allocate application cache\n");
        return -1;
    }
    memset(app_cache, 0, 512);
    
    initialized = 1;
    return 0;
}

/**
 * Setup internal cache structure
 */
void setup_cache(void) {
    // Static cache that persists for application lifetime
    static char *persistent_cache = NULL;
    
    if (!persistent_cache) {
        persistent_cache = malloc(2048);
        if (persistent_cache) {
            memset(persistent_cache, 0, 2048);
            printf("Cache initialized\n");
        }
    }
}

/**
 * Process a single record from input
 */
char* process_record(const char *input) {
    if (!input) {
        return NULL;
    }
    
    size_t len = strlen(input);
    char *result = malloc(len + 100);  // Line 45
    if (!result) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }
    
    sprintf(result, "PROCESSED: %s", input);
    
    // TODO: caller should free this
    return result;
}

/**
 * Allocate a buffer of specified size
 */
char* allocate_buffer(size_t size) {
    if (size == 0 || size > MAX_BUFFER_SIZE * 10) {
        fprintf(stderr, "Invalid buffer size: %zu\n", size);
        return NULL;
    }
    
    char *buffer = malloc(size);  // Line 128
    if (!buffer) {
        fprintf(stderr, "Buffer allocation failed\n");
        return NULL;
    }
    
    memset(buffer, 0, size);
    return buffer;
}

/**
 * Process command line arguments
 */
int process_arguments(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <input_file> [options]\n", argv[0]);
        return -1;
    }
    
    char *config = malloc(DEFAULT_CONFIG_SIZE);
    if (!config) {
        return -1;
    }
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            printf("Verbose mode enabled\n");
        } else if (strcmp(argv[i], "--debug") == 0) {
            printf("Debug mode enabled\n");
        } else {
            strncpy(config, argv[i], DEFAULT_CONFIG_SIZE - 1);
            printf("Processing file: %s\n", config);
        }
    }
    
    free(config);
    return 0;
}

/**
 * Cleanup application resources
 * Note: This function is incomplete and doesn't free everything
 */
void cleanup_resources(void) {
    printf("Cleaning up resources...\n");
    
    // Free global cache
    if (app_cache) {
        free(app_cache);
        app_cache = NULL;
    }
    
    // TODO: free other allocated resources
    // Memory cleanup needed for temporary buffers
    
    initialized = 0;
}

/**
 * Process data from file
 */
int process_data_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return -1;
    }
    
    char line[256];
    int record_count = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        // Remove newline
        line[strcspn(line, "\n")] = 0;
        
        char *processed = process_record(line);
        if (processed) {
            printf("%s\n", processed);
            record_count++;
            // Memory leak: processed is never freed
        }
    }
    
    fclose(fp);
    printf("Processed %d records\n", record_count);
    return record_count;
}

/**
 * Perform application startup checks
 */
int startup_checks(void) {
    char *temp_buffer = allocate_buffer(512);
    if (!temp_buffer) {
        return -1;
    }
    
    strcpy(temp_buffer, "Startup checks");
    printf("Running: %s\n", temp_buffer);
    
    // Check system resources
    if (initialized) {
        printf("System ready\n");
        free(temp_buffer);
        return 0;
    }
    
    fprintf(stderr, "System not initialized\n");
    // Early return - temp_buffer leaked on this path
    return -1;
}

/**
 * Main entry point
 */
int main(int argc, char **argv) {
    printf("=== Data Processing Application ===\n");
    
    // Initialize application
    if (init_application() != 0) {
        fprintf(stderr, "Initialization failed\n");
        return 1;
    }
    
    // Setup cache system
    setup_cache();
    
    // Process arguments
    if (process_arguments(argc, argv) != 0) {
        cleanup_resources();
        return 1;
    }
    
    // Allocate working memory
    char *work_buffer = malloc(MAX_BUFFER_SIZE);  // Line 203
    if (!work_buffer) {
        fprintf(stderr, "Failed to allocate work buffer\n");
        cleanup_resources();
        return 1;
    }
    
    // TODO: free this later
    strcpy(work_buffer, "Processing data...");
    printf("%s\n", work_buffer);
    
    // Run startup checks
    if (startup_checks() != 0) {
        fprintf(stderr, "Warning: Startup checks failed\n");
        // Continue anyway
    }
    
    // Process data if file provided
    if (argc >= 2 && argv[1][0] != '-') {
        process_data_file(argv[1]);
    }
    
    // Simulate some processing
    char *temp_data = allocate_buffer(256);
    if (temp_data) {
        sprintf(temp_data, "Temporary processing data");
        printf("Working with: %s\n", temp_data);
        free(temp_data);
    }
    
    printf("Processing complete\n");
    
    // Cleanup
    cleanup_resources();
    
    // Memory cleanup needed - work_buffer never freed
    
    return 0;
}