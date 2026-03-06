#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>

#define MAX_PORT 65535
#define MIN_PORT 1
#define TIMEOUT_SEC 2

/**
 * Print usage information
 */
void print_usage(const char *prog_name) {
    fprintf(stderr, "Network Utility Tool v1.0\n");
    fprintf(stderr, "Usage: %s [OPTIONS]\n\n", prog_name);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <ip>        Validate IP address format\n");
    fprintf(stderr, "  -p <port>      Check if port number is valid (1-65535)\n");
    fprintf(stderr, "  -c <ip> <port> Check if a port is open on target host\n");
    fprintf(stderr, "  -h             Display this help message\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -i 192.168.1.1\n", prog_name);
    fprintf(stderr, "  %s -p 8080\n", prog_name);
    fprintf(stderr, "  %s -c 192.168.1.1 80\n", prog_name);
}

/**
 * Validate if a string represents a valid IPv4 address
 * Returns 1 if valid, 0 otherwise
 */
int validate_ip(const char *ip_str) {
    struct sockaddr_in sa;
    int result;
    
    if (ip_str == NULL || strlen(ip_str) == 0) {
        return 0;
    }
    
    result = inet_pton(AF_INET, ip_str, &(sa.sin_addr));
    return result == 1;
}

/**
 * Validate if a port number is within valid range
 * Returns 1 if valid, 0 otherwise
 */
int validate_port(int port) {
    return (port >= MIN_PORT && port <= MAX_PORT);
}

/**
 * Check if a specific port is open on a target host
 * Returns 1 if port is open, 0 if closed, -1 on error
 */
int check_port(const char *host, int port) {
    int sockfd;
    struct sockaddr_in server_addr;
    struct timeval timeout;
    int result;
    
    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        fprintf(stderr, "Error: Failed to create socket: %s\n", strerror(errno));
        return -1;
    }
    
    // Set timeout
    timeout.tv_sec = TIMEOUT_SEC;
    timeout.tv_usec = 0;
    
    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
        fprintf(stderr, "Warning: Failed to set receive timeout\n");
    }
    
    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0) {
        fprintf(stderr, "Warning: Failed to set send timeout\n");
    }
    
    // Setup server address structure
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    // Convert IP address
    if (inet_pton(AF_INET, host, &server_addr.sin_addr) <= 0) {
        fprintf(stderr, "Error: Invalid IP address format\n");
        close(sockfd);
        return -1;
    }
    
    // Attempt connection
    result = connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr));
    
    close(sockfd);
    
    if (result == 0) {
        return 1; // Port is open
    } else {
        return 0; // Port is closed or unreachable
    }
}

/**
 * Main function - process command line arguments and execute requested operations
 */
int main(int argc, char *argv[]) {
    int opt;
    int port;
    char *ip_address = NULL;
    
    // Check if no arguments provided
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    
    // Parse command line options
    while ((opt = getopt(argc, argv, "i:p:c:h")) != -1) {
        switch (opt) {
            case 'i':
                // Validate IP address
                ip_address = optarg;
                if (validate_ip(ip_address)) {
                    printf("IP Address %s: VALID\n", ip_address);
                    return EXIT_SUCCESS;
                } else {
                    fprintf(stderr, "IP Address %s: INVALID\n", ip_address);
                    return EXIT_FAILURE;
                }
                break;
                
            case 'p':
                // Validate port number
                port = atoi(optarg);
                if (validate_port(port)) {
                    printf("Port %d: VALID (within range %d-%d)\n", 
                           port, MIN_PORT, MAX_PORT);
                    return EXIT_SUCCESS;
                } else {
                    fprintf(stderr, "Port %d: INVALID (must be between %d and %d)\n",
                           port, MIN_PORT, MAX_PORT);
                    return EXIT_FAILURE;
                }
                break;
                
            case 'c':
                // Check port connectivity
                if (optind >= argc) {
                    fprintf(stderr, "Error: -c option requires both IP and port\n");
                    print_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                
                ip_address = optarg;
                port = atoi(argv[optind]);
                
                // Validate inputs
                if (!validate_ip(ip_address)) {
                    fprintf(stderr, "Error: Invalid IP address: %s\n", ip_address);
                    return EXIT_FAILURE;
                }
                
                if (!validate_port(port)) {
                    fprintf(stderr, "Error: Invalid port number: %d\n", port);
                    return EXIT_FAILURE;
                }
                
                printf("Checking connection to %s:%d...\n", ip_address, port);
                
                int status = check_port(ip_address, port);
                if (status == 1) {
                    printf("Port %d on %s: OPEN\n", port, ip_address);
                    return EXIT_SUCCESS;
                } else if (status == 0) {
                    printf("Port %d on %s: CLOSED or FILTERED\n", port, ip_address);
                    return EXIT_FAILURE;
                } else {
                    fprintf(stderr, "Error: Failed to check port connectivity\n");
                    return EXIT_FAILURE;
                }
                break;
                
            case 'h':
                // Display help
                print_usage(argv[0]);
                return EXIT_SUCCESS;
                
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }
    
    // If we reach here, no valid option was processed
    print_usage(argv[0]);
    return EXIT_FAILURE;
}