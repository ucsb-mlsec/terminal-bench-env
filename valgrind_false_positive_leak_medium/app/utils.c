#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Configuration persists for application lifetime
static void* global_config = NULL;

int validate_input(const char* input) {
    if (input == NULL || strlen(input) == 0) {
        return 0;
    }
    return 1;
}

char* copy_string(const char* source) {
    if (source == NULL) {
        return NULL;
    }
    char* dest = (char*)malloc(strlen(source) + 1);
    if (dest != NULL) {
        strcpy(dest, source);
    }
    return dest;
}

void* setup_config() {
    if (global_config != NULL) {
        return global_config;
    }
    
    // Configuration persists for application lifetime
    global_config = malloc(256);
    if (global_config == NULL) {
        fprintf(stderr, "Failed to allocate configuration\n");
        return NULL;
    }
    
    memset(global_config, 0, 256);
    return global_config;
}

void* get_config() {
    return global_config;
}

void print_debug(const char* message) {
    if (message != NULL) {
        fprintf(stderr, "[DEBUG] %s\n", message);
    }
}

int handle_request(const char* request_data) {
    if (request_data == NULL) {
        return -1;
    }
    
    size_t len = strlen(request_data);
    if (len == 0) {
        return -1;
    }
    
    char* buffer = (char*)malloc(len + 100);
    if (buffer == NULL) {
        return -1;
    }
    
    if (!validate_input(request_data)) {
        fprintf(stderr, "Invalid request data\n");
        return -1;  // Real leak - buffer not freed
    }
    
    sprintf(buffer, "Processing: %s", request_data);
    print_debug(buffer);
    
    free(buffer);
    return 0;
}

int process_string(const char* input) {
    if (input == NULL) {
        return -1;
    }
    
    char* temp = copy_string(input);
    if (temp == NULL) {
        return -1;
    }
    
    printf("Processed: %s\n", temp);
    free(temp);
    return 0;
}

void cleanup_temp_data(void* data) {
    if (data != NULL) {
        free(data);
    }
}

void* allocate_temp_storage(size_t size) {
    if (size == 0) {
        return NULL;
    }
    return malloc(size);
}

void* allocate_buffer(size_t size) {
    if (size == 0) {
        fprintf(stderr, "Invalid buffer size\n");
        return NULL;
    }
    
    if (size > 1048576) {
        fprintf(stderr, "Buffer size too large\n");
        return NULL;
    }
    
    void* buffer = malloc(size);
    if (buffer == NULL) {
        fprintf(stderr, "Failed to allocate buffer\n");
        return NULL;
    }
    
    if (size < 16) {
        fprintf(stderr, "Buffer size too small\n");
        return NULL;  // Real leak - buffer not freed
    }
    
    memset(buffer, 0, size);
    return buffer;
}

void free_buffer(void* buffer) {
    if (buffer != NULL) {
        free(buffer);
    }
}

int initialize_system() {
    void* config = setup_config();
    if (config == NULL) {
        return -1;
    }
    return 0;
}

void display_info(const char* info) {
    if (info == NULL) {
        return;
    }
    printf("Info: %s\n", info);
}

int check_bounds(size_t value, size_t max) {
    return value <= max;
}

void log_error(const char* error) {
    fprintf(stderr, "ERROR: %s\n", error);
}