#include <stdio.h>
#include <string.h>
#include "udp_handler.h"

int udp_packet_count = 0;
int udp_error_count = 0;

void init_udp_handler() {
    printf("Initializing UDP handler...\n");
    udp_packet_count = 0;
    udp_error_count = 0;
    printf("UDP handler initialized successfully\n");
}

void process_udp_packet(const char* data, int len) {
    if (data == NULL || len <= 0) {
        printf("Error: Invalid UDP packet data\n");
        udp_error_count++;
        return;
    }
    
    printf("Processing UDP packet (length: %d bytes)\n", len);
    udp_packet_count++;
    
    // Basic UDP packet validation
    if (len < 8) {
        printf("Warning: UDP packet too small (minimum 8 bytes required)\n");
        udp_error_count++;
        return;
    }
    
    // Simulate packet processing
    printf("UDP packet processed successfully\n");
    printf("Source port: %d, Destination port: %d\n", 
           (int)data[0] * 256 + (int)data[1],
           (int)data[2] * 256 + (int)data[3]);
}

int get_udp_packet_count() {
    return udp_packet_count;
}

void print_udp_statistics() {
    printf("\n=== UDP Handler Statistics ===\n");
    printf("Total UDP packets processed: %d\n", udp_packet_count);
    printf("UDP errors encountered: %d\n", udp_error_count);
    printf("==============================\n\n");
}

void reset_udp_handler() {
    printf("Resetting UDP handler statistics...\n");
    udp_packet_count = 0;
    udp_error_count = 0;
}