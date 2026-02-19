#ifndef NETWORK_H
#define NETWORK_H

#include <stdio.h>
#include <stdint.h>

int connect_to_server(const char *hostname, int port);
int send_data(int socket_fd, const char *data, size_t length);
int receive_data(int socket_fd, char *buffer, size_t buffer_size);
void close_connection(int socket_fd);

#endif