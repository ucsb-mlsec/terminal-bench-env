#include "processor.h"
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <netinet/in.h>

#define MAX_THREADS 10
#define BUFFER_SIZE 4096
#define PORT 8080

/* Global thread pool */
static pthread_t thread_pool[MAX_THREADS];
static int active_threads = 0;

/**
 * validate_input - Validates incoming request data
 * @data: Pointer to data buffer
 * @size: Size of data
 * 
 * Returns: 0 on success, -1 on failure
 */
int validate_input(const char *data, size_t size) {
    if (!data || size == 0) {
        fprintf(stderr, "Invalid input data\n");
        return -1;
    }
    
    if (size > BUFFER_SIZE) {
        fprintf(stderr, "Input data exceeds buffer size\n");
        return -1;
    }
    
    return 0;
}

/**
 * parse_data - Parses the payload data
 * @payload: Data to parse
 * @len: Length of payload
 * 
 * Returns: Parsed data structure or NULL on error
 */
void *parse_data(const char *payload, size_t len) {
    if (!payload || len == 0) {
        return NULL;
    }
    
    /* Simple parsing logic */
    void *parsed = malloc(len);
    if (parsed) {
        memcpy(parsed, payload, len);
    }
    
    return parsed;
}

/**
 * handle_connection - Handles an active connection
 * @conn: Pointer to connection structure
 * 
 * Returns: 0 on success, -1 on error
 */
int handle_connection(connection_t *conn) {
    if (!conn) {
        fprintf(stderr, "NULL connection pointer\n");
        return -1;
    }
    
    /* Attempt to read from buffer */
    if (conn->buffer && conn->buffer_size > 0) {
        printf("Processing buffer of size %zu\n", conn->buffer_size);
        
        /* Simulate processing */
        for (size_t i = 0; i < conn->buffer_size && i < 100; i++) {
            conn->buffer[i] = conn->buffer[i] ^ 0xFF;
        }
        
        return 0;
    }
    
    fprintf(stderr, "Invalid buffer in connection\n");
    return -1;
}

/**
 * create_connection - Allocates and initializes a connection structure
 * @socket_fd: File descriptor for the socket
 * 
 * Returns: Pointer to new connection or NULL on error
 */
connection_t *create_connection(int socket_fd) {
    connection_t *conn = (connection_t *)malloc(sizeof(connection_t));
    if (!conn) {
        fprintf(stderr, "Failed to allocate connection structure\n");
        return NULL;
    }
    
    conn->socket_fd = socket_fd;
    conn->buffer = (char *)malloc(BUFFER_SIZE);
    if (!conn->buffer) {
        fprintf(stderr, "Failed to allocate connection buffer\n");
        free(conn);
        return NULL;
    }
    
    conn->buffer_size = BUFFER_SIZE;
    conn->is_active = 1;
    
    return conn;
}

/**
 * process_request - Main request processing function
 * @arg: Void pointer to request_t structure
 * 
 * This function is executed by worker threads to process incoming requests.
 * Returns: NULL
 */
void *process_request(void *arg) {
    request_t *req = (request_t *)arg;
    
    if (!req) {
        fprintf(stderr, "NULL request received\n");
        return NULL;
    }
    
    printf("Processing request from client\n");
    
    /* Validate the input data */
    if (validate_input(req->data, req->data_size) != 0) {
        fprintf(stderr, "Request validation failed\n");
        free(req->data);
        free(req);
        return NULL;
    }
    
    /* Parse the payload */
    void *parsed = parse_data(req->data, req->data_size);
    if (!parsed) {
        fprintf(stderr, "Failed to parse request data\n");
        free(req->data);
        free(req);
        return NULL;
    }
    
    /* Get the connection pointer from the request */
    connection_t *conn = req->conn;
    
    /* BUG: Accessing conn->buffer without checking if conn is NULL
     * This is the critical bug that causes the segmentation fault
     * In some cases, req->conn might be NULL due to race conditions
     * during connection cleanup or initialization errors */
    size_t buf_len = strlen(conn->buffer);
    
    if (buf_len > 0) {
        /* Handle the connection */
        if (handle_connection(conn) != 0) {
            fprintf(stderr, "Connection handling failed\n");
        }
    }
    
    /* Cleanup */
    free(parsed);
    free(req->data);
    free(req);
    
    return NULL;
}

/**
 * main - Entry point for the server application
 * 
 * Sets up the server socket, thread pool, and accepts incoming connections.
 */
int main(int argc, char *argv[]) {
    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    printf("Starting request processor server on port %d\n", PORT);
    
    /* Create server socket */
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }
    
    /* Configure server address */
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);
    
    /* Bind socket */
    if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    
    /* Listen for connections */
    if (listen(server_fd, 10) < 0) {
        perror("Listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    
    printf("Server listening for connections...\n");
    
    /* Main accept loop */
    while (1) {
        client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("Accept failed");
            continue;
        }
        
        printf("Accepted new connection\n");
        
        /* Create request structure and spawn thread */
        request_t *req = (request_t *)malloc(sizeof(request_t));
        if (req) {
            req->data = (char *)malloc(BUFFER_SIZE);
            req->data_size = read(client_fd, req->data, BUFFER_SIZE);
            req->conn = create_connection(client_fd);
            
            if (active_threads < MAX_THREADS) {
                pthread_create(&thread_pool[active_threads++], NULL, process_request, req);
            }
        }
    }
    
    close(server_fd);
    return 0;
}