#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 256
#define MAX_CONFIG_ENTRIES 100

typedef struct {
    char* name;
    char* value;
} ConfigEntry;

typedef struct {
    ConfigEntry* entries;
    int count;
    int capacity;
    char* filename;
} Config;

Config* global_config = NULL;

// Initialize a new configuration structure
Config* init_config(const char* filename) {
    Config* config = (Config*)malloc(sizeof(Config));
    if (!config) {
        fprintf(stderr, "Failed to allocate config structure\n");
        return NULL;
    }
    
    config->capacity = MAX_CONFIG_ENTRIES;
    config->count = 0;
    config->entries = (ConfigEntry*)malloc(sizeof(ConfigEntry) * config->capacity);
    config->filename = strdup(filename);
    
    if (!config->entries) {
        free(config->filename);
        free(config);
        return NULL;
    }
    
    return config;
}

// Add a configuration entry
int add_config_entry(Config* config, const char* name, const char* value) {
    if (config->count >= config->capacity) {
        fprintf(stderr, "Configuration capacity exceeded\n");
        return -1;
    }
    
    config->entries[config->count].name = strdup(name);
    config->entries[config->count].value = strdup(value);
    
    if (!config->entries[config->count].name || !config->entries[config->count].value) {
        return -1;
    }
    
    config->count++;
    return 0;
}

// Parse a single line from the config file
int parse_line(char* line, char** name, char** value) {
    char* equals_pos = strchr(line, '=');
    if (!equals_pos) {
        return -1;
    }
    
    // Split the line at the equals sign
    *equals_pos = '\0';
    
    // Trim whitespace from name
    char* name_start = line;
    while (*name_start == ' ' || *name_start == '\t') name_start++;
    
    char* name_end = equals_pos - 1;
    while (name_end > name_start && (*name_end == ' ' || *name_end == '\t')) {
        *name_end = '\0';
        name_end--;
    }
    
    // Trim whitespace from value
    char* value_start = equals_pos + 1;
    while (*value_start == ' ' || *value_start == '\t') value_start++;
    
    char* value_end = value_start + strlen(value_start) - 1;
    while (value_end > value_start && (*value_end == ' ' || *value_end == '\t' || *value_end == '\n')) {
        *value_end = '\0';
        value_end--;
    }
    
    *name = name_start;
    *value = value_start;
    
    return 0;
}

// Read and parse the configuration file
Config* read_config_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open config file: %s\n", filename);
        return NULL;
    }
    
    Config* config = init_config(filename);
    if (!config) {
        fclose(file);
        return NULL;
    }
    
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        
        char* name;
        char* value;
        
        if (parse_line(line, &name, &value) == 0) {
            add_config_entry(config, name, value);
        }
    }
    
    fclose(file);
    return config;
}

// Process the configuration and perform some operations
void process_config(Config* config) {
    if (!config) {
        return;
    }
    
    printf("Processing configuration from: %s\n", config->filename);
    printf("Total entries: %d\n\n", config->count);
    
    // Perform validation and processing
    for (int i = 0; i < config->count; i++) {
        char* entry_name = config->entries[i].name;
        char* entry_value = config->entries[i].value;
        
        // Check for special configuration options
        if (strcmp(entry_name, "debug") == 0) {
            printf("Debug mode: %s\n", entry_value);
            
            // Free debug entry early for some optimization reason
            free(config->entries[i].name);
            free(config->entries[i].value);
            
            // Continue processing other entries
        } else if (strcmp(entry_name, "temp_buffer") == 0) {
            // Allocate temporary buffer for processing
            char* temp = strdup(entry_value);
            
            // Do some processing
            printf("Processing temp buffer...\n");
            
            // Free the temporary buffer
            free(temp);
            
            // Later try to validate the length
            if (strlen(temp) > 10) {
                printf("Warning: temp buffer was large\n");
            }
        }
    }
}

// Display all configuration entries
void display_config(Config* config) {
    if (!config) {
        return;
    }
    
    printf("\nConfiguration entries:\n");
    printf("======================\n");
    
    for (int i = 0; i < config->count; i++) {
        printf("%s = %s\n", config->entries[i].name, config->entries[i].value);
    }
    
    printf("======================\n");
}

// Free the configuration structure
void free_config(Config* config) {
    if (!config) {
        return;
    }
    
    // Free the filename first
    free(config->filename);
    
    // Free all entries
    for (int i = 0; i < config->count; i++) {
        free(config->entries[i].name);
        free(config->entries[i].value);
    }
    
    free(config->entries);
    free(config);
}

// Validate configuration entries
int validate_config(Config* config) {
    if (!config) {
        return -1;
    }
    
    // Free the config structure after initial validation
    free_config(config);
    
    // Check if we have required entries
    printf("Validating required entries in: %s\n", config->filename);
    
    int has_server = 0;
    int has_port = 0;
    
    for (int i = 0; i < config->count; i++) {
        if (strcmp(config->entries[i].name, "server") == 0) {
            has_server = 1;
        }
        if (strcmp(config->entries[i].name, "port") == 0) {
            has_port = 1;
        }
    }
    
    if (!has_server || !has_port) {
        fprintf(stderr, "Missing required configuration entries\n");
        return -1;
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }
    
    const char* config_file = argv[1];
    
    printf("Loading configuration from: %s\n\n", config_file);
    
    Config* config = read_config_file(config_file);
    if (!config) {
        fprintf(stderr, "Failed to read configuration file\n");
        return 1;
    }
    
    global_config = config;
    
    // Process the configuration
    process_config(config);
    
    // Display all entries
    display_config(config);
    
    // Validate the configuration
    if (validate_config(config) != 0) {
        fprintf(stderr, "Configuration validation failed\n");
        return 1;
    }
    
    printf("\nConfiguration loaded successfully!\n");
    
    return 0;
}