#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <json-c/json.h>

void print_help() {
    printf("Usage: config_parser [options]\n");
    printf("  --help    Show this help message\n");
    printf("  --config  Parse configuration file (default: config.json)\n");
    printf("\nThis utility parses JSON configuration files.\n");
}

int parse_config_string(const char *json_str) {
    struct json_object *parsed_json;
    struct json_object *name;
    struct json_object *version;
    struct json_object *enabled;
    
    parsed_json = json_tokener_parse(json_str);
    
    if (parsed_json == NULL) {
        fprintf(stderr, "Error: Failed to parse JSON string\n");
        return 1;
    }
    
    if (json_object_object_get_ex(parsed_json, "app_name", &name)) {
        printf("Application Name: %s\n", json_object_get_string(name));
    }
    
    if (json_object_object_get_ex(parsed_json, "version", &version)) {
        printf("Version: %s\n", json_object_get_string(version));
    }
    
    if (json_object_object_get_ex(parsed_json, "enabled", &enabled)) {
        printf("Enabled: %s\n", json_object_get_boolean(enabled) ? "true" : "false");
    }
    
    json_object_put(parsed_json);
    return 0;
}

int parse_config_file(const char *filename) {
    FILE *fp;
    char buffer[4096];
    size_t bytes_read;
    
    fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Warning: Could not open config file '%s', using defaults\n", filename);
        return 1;
    }
    
    bytes_read = fread(buffer, 1, sizeof(buffer) - 1, fp);
    buffer[bytes_read] = '\0';
    fclose(fp);
    
    printf("Parsing configuration from file: %s\n", filename);
    return parse_config_string(buffer);
}

int main(int argc, char *argv[]) {
    const char *default_config = "{\"app_name\": \"config_parser\", \"version\": \"1.0.0\", \"enabled\": true}";
    
    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        print_help();
        return 0;
    }
    
    printf("JSON Configuration Parser v1.0\n");
    printf("==============================\n\n");
    
    if (argc > 1 && strcmp(argv[1], "--config") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Error: --config requires a filename argument\n");
            return 1;
        }
        return parse_config_file(argv[2]);
    }
    
    printf("Parsing default configuration:\n");
    int result = parse_config_string(default_config);
    
    if (result == 0) {
        printf("\nConfiguration parsed successfully!\n");
    }
    
    return result;
}