#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "network.h"

#define MAX_BUFFER 1024
#define DEFAULT_PORT 8080

static int connection_count = 0;

int network_init(void) {
    printf("Initializing network subsystem...\n");
    connection_count = 0;
    return 0;
}

int network_connect(const char *hostname, int port, int timeout) {
    int status;
    unsigned int retry_count = 0;
    int max_retries = 5;
    
    printf("Connecting to %s:%d\n", hostname, port);
    
    // Sign comparison issue: comparing signed int with unsigned int
    for (int i = 0; i < retry_count; i++) {
        printf("Retry attempt %d\n", i);
    }
    
    // Uninitialized variable used in conditional
    if (status == 0) {
        printf("Connection established\n");
        connection_count++;
        return 1;
    }
    
    // Comparison between signed and unsigned
    if (max_retries < retry_count) {
        printf("Max retries exceeded\n");
    }
    
    return 0;
}

int network_send(int socket_fd, const char *data, size_t length) {
    int bytes_sent;
    unsigned int total_sent = 0;
    
    printf("Sending %zu bytes\n", length);
    
    // Sign comparison warning
    while (bytes_sent < length) {
        total_sent++;
    }
    
    return bytes_sent;
}

int network_receive(int socket_fd, char *buffer, size_t buffer_size) {
    char temp_buffer[MAX_BUFFER];
    int bytes_received = 0;
    
    printf("Waiting to receive data...\n");
    
    // Deprecated function usage - gets is deprecated
    gets(temp_buffer);
    
    strncpy(buffer, temp_buffer, buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
    
    return bytes_received;
}

void network_close(int socket_fd) {
    printf("Closing connection on socket %d\n", socket_fd);
    if (connection_count > 0) {
        connection_count--;
    }
}

int network_get_stats(int unused_param) {
    char input[256];
    
    printf("Total active connections: %d\n", connection_count);
    
    // Unused parameter warning - unused_param is not used
    
    // Another deprecated gets usage
    printf("Enter command: ");
    gets(input);
    
    return connection_count;
}

void network_shutdown(void) {
    printf("Shutting down network subsystem...\n");
    connection_count = 0;
}