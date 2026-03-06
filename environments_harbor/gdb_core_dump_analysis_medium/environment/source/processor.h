#ifndef PROCESSOR_H
#define PROCESSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

/* Configuration constants */
#define MAX_BUFFER_SIZE 1024
#define MAX_CONNECTIONS 100

/* Connection structure - represents a single client connection */
typedef struct connection_t {
    int socket_fd;          /* File descriptor for the socket */
    char *buffer;           /* Buffer for incoming data */
    int buffer_size;        /* Current size of allocated buffer */
    int is_active;          /* Connection active flag */
    void *user_data;        /* Additional connection-specific data */
} connection_t;

/* Request structure - represents a parsed client request */
typedef struct request_t {
    char *payload;          /* Request payload data */
    int length;             /* Length of payload */
    connection_t *conn;     /* Associated connection */
    long timestamp;         /* Request timestamp */
} request_t;

/* Thread pool structure - manages worker threads */
typedef struct thread_pool_t {
    pthread_t *threads;     /* Array of thread handles */
    int thread_count;       /* Number of threads in pool */
    int active;             /* Pool active status */
} thread_pool_t;

/* Function prototypes */

/* Main request processing function - runs in worker threads */
void* process_request(void *arg);

/* Handle a single connection - processes incoming data */
int handle_connection(connection_t *conn);

/* Parse request data into usable format */
char* parse_data(request_t *req);

/* Validate input data for correctness */
int validate_input(char *data, int len);

/* Clean up connection resources */
void cleanup_resources(connection_t *conn);

/* Create a new connection structure */
connection_t* create_connection(int fd);

#endif