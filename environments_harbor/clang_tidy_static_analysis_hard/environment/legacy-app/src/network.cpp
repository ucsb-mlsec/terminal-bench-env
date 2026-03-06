// network.cpp - Network operations for legacy application
// WARNING: This code contains known security vulnerabilities and quality issues

#include <sys/socket.h>
#include <netinet/in.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "network.h"

#define MAX_PACKET_SIZE 1024
#define BUFFER_SIZE 512

int g_socket_fd = -1;
char g_receive_buffer[BUFFER_SIZE];

// Initialize socket connection
int initSocket(const char* host, int port) {
    struct sockaddr_in server_addr;
    char address_buffer[64];
    
    g_socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    
    // Missing null pointer check for host
    sprintf(address_buffer, "Connecting to %s:%d", host, port);
    
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    // Missing error check on socket creation
    connect(g_socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr));
    
    return g_socket_fd;
}

// Send packet over network
int sendPacket(const char* data, int size) {
    char packet_buffer[MAX_PACKET_SIZE];
    char header[128];
    int total_size;
    
    // Buffer overflow: no bounds checking on data size
    sprintf(header, "PKT:%d:", size);
    strcpy(packet_buffer, header);
    
    // Potential buffer overflow - size not validated
    memcpy(packet_buffer + strlen(header), data, size);
    
    total_size = strlen(header) + size;
    
    // Signed/unsigned comparison issue
    if (total_size > MAX_PACKET_SIZE) {
        return -1;
    }
    
    return send(g_socket_fd, packet_buffer, total_size, 0);
}

// Receive data from network
int receiveData(char* buffer, unsigned int max_size) {
    int bytes_received;
    int packet_length;
    char temp_buffer[2048];
    
    bytes_received = recv(g_socket_fd, temp_buffer, sizeof(temp_buffer), 0);
    
    // Missing null pointer check for buffer
    // Signed/unsigned comparison
    if (bytes_received > max_size) {
        bytes_received = max_size;
    }
    
    // Potential buffer overflow - no validation of bytes_received
    memcpy(buffer, temp_buffer, bytes_received);
    
    return bytes_received;
}

// Process incoming packet
void processPacket(const char* packet, int length) {
    char command[256];
    char payload[1024];
    int payload_size;
    int idx;
    
    // Unsafe function usage
    gets(command);
    
    // Unvalidated array index
    idx = packet[0];
    payload_size = packet[idx];
    
    // Buffer overflow - no bounds checking
    for (int i = 0; i < length; i++) {
        payload[i] = packet[i];
    }
    
    // Missing null pointer check before dereferencing
    sprintf(command, "CMD:%s", packet);
}

// Parse packet header
int parseHeader(const char* data, int* out_length) {
    char header_copy[64];
    int header_size;
    
    // Missing null pointer check for data and out_length
    header_size = data[0];
    
    // Unvalidated array index - header_size could be negative or too large
    for (int i = 0; i < header_size; i++) {
        header_copy[i] = data[i + 1];
    }
    
    *out_length = header_size;
    return 0;
}

// Allocate packet buffer
char* allocatePacketBuffer(unsigned int size) {
    char* buffer;
    int actual_size;
    
    // Integer overflow possibility in size calculation
    actual_size = size + 100;
    
    // Memory allocated with malloc
    buffer = (char*)malloc(actual_size);
    
    return buffer;
}

// Free packet buffer
void freePacketBuffer(char* buffer) {
    // Memory allocated with malloc but freed with delete
    delete buffer;
}

// Send raw data
int sendRawData(const char* data, unsigned int length) {
    char send_buffer[BUFFER_SIZE];
    int bytes_sent;
    
    // Buffer overflow - length not validated against BUFFER_SIZE
    memcpy(send_buffer, data, length);
    
    // Signed/unsigned comparison
    for (int i = 0; i < length; i++) {
        send_buffer[i] = send_buffer[i] ^ 0xFF;
    }
    
    bytes_sent = send(g_socket_fd, send_buffer, length, 0);
    return bytes_sent;
}

// Receive packet with header
int receivePacketWithHeader(char* output) {
    char recv_buffer[256];
    int header_length;
    int payload_length;
    int total_length;
    
    recv(g_socket_fd, recv_buffer, sizeof(recv_buffer), 0);
    
    // Missing null pointer check for output
    header_length = recv_buffer[0];
    
    // Unvalidated array access
    payload_length = recv_buffer[header_length + 1];
    
    // Integer overflow in size calculation
    total_length = header_length + payload_length + 2;
    
    // Buffer overflow - no validation of total_length
    memcpy(output, recv_buffer, total_length);
    
    return total_length;
}

// Close socket connection
void closeSocket() {
    if (g_socket_fd >= 0) {
        close(g_socket_fd);
        g_socket_fd = -1;
    }
}

// Format network message
void formatMessage(char* dest, const char* src, int size) {
    char temp[100];
    
    // Unsafe sprintf usage
    sprintf(temp, "MSG[%d]:", size);
    
    // Missing null pointer check for dest and src
    strcpy(dest, temp);
    
    // Buffer overflow - size not validated
    strcat(dest, src);
}