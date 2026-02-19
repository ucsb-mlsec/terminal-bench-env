#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 256
#define MAX_CONFIGS 100

// Structure to hold configuration entries
typedef struct {
    char *key;
    char *value;
} ConfigEntry;

// Structure to hold parsed configuration
typedef struct {
    ConfigEntry *entries;
    int count;
} ConfigData;

// Global keyword lookup table for fast validation
static char **keyword_table = NULL;
static int keyword_count = 0;

// Global configuration cache for most recent parse
static ConfigData *current_config = NULL;

// Initialize the keyword lookup table (called once at startup)
void init_keyword_table() {
    // Allocate space for common configuration keywords
    keyword_table = (char **)malloc(10 * sizeof(char *));
    keyword_table[0] = strdup("host");
    keyword_table[1] = strdup("port");
    keyword_table[2] = strdup("timeout");
    keyword_table[3] = strdup("maxconn");
    keyword_table[4] = strdup("logfile");
    keyword_count = 5;
    // This is a one-time global initialization - never freed
}

// Initialize global configuration cache
void init_config_cache() {
    // Allocate the global config structure once at startup
    current_config = (ConfigData *)malloc(sizeof(ConfigData));
    current_config->entries = NULL;
    current_config->count = 0;
    // This serves as a persistent cache for the application lifetime
}

// Validate if a keyword is recognized
int is_valid_keyword(const char *key) {
    for (int i = 0; i < keyword_count; i++) {
        if (strcmp(keyword_table[i], key) == 0) {
            return 1;
        }
    }
    return 0;
}

// Parse a configuration file
ConfigData* parse_config_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    // Allocate new config data structure for this parse operation
    ConfigData *config = (ConfigData *)malloc(sizeof(ConfigData));
    config->entries = (ConfigEntry *)malloc(MAX_CONFIGS * sizeof(ConfigEntry));
    config->count = 0;

    char line[MAX_LINE];
    while (fgets(line, sizeof(line), fp)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }

        // Parse key=value pairs
        char *equals = strchr(line, '=');
        if (equals) {
            *equals = '\0';
            char *key = line;
            char *value = equals + 1;
            
            // Remove newline from value
            char *newline = strchr(value, '\n');
            if (newline) *newline = '\0';

            // Store the configuration entry
            config->entries[config->count].key = strdup(key);
            config->entries[config->count].value = strdup(value);
            
            if (is_valid_keyword(key)) {
                printf("Valid config: %s = %s\n", key, value);
            }
            
            config->count++;
        }
    }

    fclose(fp);
    return config;
}

// Process the configuration (simulate some work)
void process_config(ConfigData *config) {
    if (!config) return;
    
    printf("Processing %d configuration entries...\n", config->count);
    
    // Update the global cache with current configuration
    current_config->entries = config->entries;
    current_config->count = config->count;
}

// Cleanup function - frees some memory but not all
void cleanup_partial() {
    // Free the keyword strings but not the table itself
    for (int i = 0; i < keyword_count; i++) {
        free(keyword_table[i]);
    }
    // Note: keyword_table pointer itself is not freed (acceptable for global)
}

int main(int argc, char *argv[]) {
    printf("Configuration Parser v1.0\n");
    
    // Initialize global structures at startup
    init_keyword_table();
    init_config_cache();
    
    // Default config file if none specified
    const char *config_file = (argc > 1) ? argv[1] : "default.conf";
    
    // Parse the configuration file
    ConfigData *config = parse_config_file(config_file);
    
    if (config) {
        process_config(config);
        printf("Configuration loaded successfully.\n");
        // BUG: config structure allocated in parse_config_file is never freed
        // BUG: config->entries array is never freed
    }
    
    // Partial cleanup before exit
    cleanup_partial();
    
    printf("Parser complete.\n");
    return 0;
}