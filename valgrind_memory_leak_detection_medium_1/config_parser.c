#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 256
#define MAX_ENTRIES 100

typedef struct {
    char *key;
    char *value;
} ConfigEntry;

typedef struct {
    ConfigEntry *entries;
    int count;
} ConfigData;

char* trim_whitespace(char *str) {
    char *end;
    while(isspace((unsigned char)*str)) str++;
    if(*str == 0) return str;
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

int parse_line(char *line, char **key, char **value) {
    char *trimmed = trim_whitespace(line);
    
    if (strlen(trimmed) == 0 || trimmed[0] == '#') {
        return 0;
    }
    
    char *equals = strchr(trimmed, '=');
    if (equals == NULL) {
        return 0;
    }
    
    *equals = '\0';
    char *k = trim_whitespace(trimmed);
    char *v = trim_whitespace(equals + 1);
    
    if (strlen(k) == 0) {
        return 0;
    }
    
    *key = strdup(k);
    *value = strdup(v);
    
    return 1;
}

ConfigData* parse_config_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        return NULL;
    }
    
    ConfigData *config = (ConfigData*)malloc(sizeof(ConfigData));
    config->entries = (ConfigEntry*)malloc(sizeof(ConfigEntry) * MAX_ENTRIES);
    config->count = 0;
    
    char *buffer = (char*)malloc(MAX_LINE);
    
    while (fgets(buffer, MAX_LINE, file) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len-1] == '\n') {
            buffer[len-1] = '\0';
        }
        
        char *key = NULL;
        char *value = NULL;
        
        if (parse_line(buffer, &key, &value)) {
            if (config->count < MAX_ENTRIES) {
                config->entries[config->count].key = key;
                config->entries[config->count].value = value;
                config->count++;
            }
        }
    }
    
    return config;
}

void print_config(ConfigData *config) {
    printf("Configuration Settings:\n");
    printf("=======================\n");
    for (int i = 0; i < config->count; i++) {
        printf("%s = %s\n", config->entries[i].key, config->entries[i].value);
    }
    printf("=======================\n");
    printf("Total entries: %d\n", config->count);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }
    
    ConfigData *config = parse_config_file(argv[1]);
    
    if (config == NULL) {
        return 1;
    }
    
    print_config(config);
    
    return 0;
}