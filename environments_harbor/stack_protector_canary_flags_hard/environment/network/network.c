#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

int create_socket(int port) {
    int sockfd;
    struct sockaddr_in addr;
    
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return -1;
    }
    
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Bind failed");
        close(sockfd);
        return -1;
    }
    
    return sockfd;
}

int send_data(int sockfd, const char* buffer, size_t len) {
    ssize_t bytes_sent;
    
    bytes_sent = send(sockfd, buffer, len, 0);
    if (bytes_sent < 0) {
        perror("Send failed");
        return -1;
    }
    
    return bytes_sent;
}

int receive_data(int sockfd, char* buffer, size_t max_len) {
    ssize_t bytes_received;
    
    memset(buffer, 0, max_len);
    bytes_received = recv(sockfd, buffer, max_len - 1, 0);
    
    if (bytes_received < 0) {
        perror("Receive failed");
        return -1;
    }
    
    return bytes_received;
}