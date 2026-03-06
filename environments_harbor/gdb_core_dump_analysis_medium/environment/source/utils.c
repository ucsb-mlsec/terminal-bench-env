#include "processor.h"
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

char* parse_data(request_t* req) {
    if (req == NULL || req->payload == NULL) {
        log_message("ERROR: NULL request or payload in parse_data");
        return NULL;
    }
    
    size_t len = strlen(req->payload);
    if (len == 0 || len > MAX_PAYLOAD_SIZE) {
        log_message("ERROR: Invalid payload length");
        return NULL;
    }
    
    char* result = (char*)malloc(len + 1);
    if (result == NULL) {
        log_message("ERROR: Memory allocation failed in parse_data");
        return NULL;
    }
    
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        if (isalnum(req->payload[i]) || req->payload[i] == ' ' || req->payload[i] == '-') {
            result[j++] = req->payload[i];
        }
    }
    result[j] = '\0';
    
    return result;
}

void cleanup_resources(connection_t* conn) {
    if (conn == NULL) {
        return;
    }
    
    if (conn->request != NULL) {
        if (conn->request->payload != NULL) {
            free(conn->request->payload);
            conn->request->payload = NULL;
        }
        free(conn->request);
        conn->request = NULL;
    }
    
    if (conn->socket_fd > 0) {
        close(conn->socket_fd);
        conn->socket_fd = -1;
    }
    
    conn->active = 0;
}

void log_message(const char* message) {
    if (message == NULL) {
        return;
    }
    
    char* timestamp = get_timestamp();
    if (timestamp != NULL) {
        fprintf(stderr, "[%s] %s\n", timestamp, message);
        free(timestamp);
    } else {
        fprintf(stderr, "%s\n", message);
    }
}

char* get_timestamp() {
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    
    char* buffer = (char*)malloc(32);
    if (buffer != NULL) {
        strftime(buffer, 32, "%Y-%m-%d %H:%M:%S", tm_info);
    }
    return buffer;
}

char* sanitize_input(const char* input) {
    if (input == NULL) {
        return NULL;
    }
    
    size_t len = strlen(input);
    char* sanitized = (char*)malloc(len + 1);
    if (sanitized == NULL) {
        return NULL;
    }
    
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        if (input[i] != '<' && input[i] != '>' && 
            input[i] != '&' && input[i] != '"' &&
            input[i] != '\'' && input[i] != ';') {
            sanitized[j++] = input[i];
        }
    }
    sanitized[j] = '\0';
    
    return sanitized;
}