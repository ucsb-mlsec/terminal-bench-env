#include <stdio.h>
#include "tcp_handler.h"

int tcp_packet_count = 0;
int tcp_error_count = 0;

void init_tcp_handler() {
    printf("Initializing TCP handler...\n");
    tcp_packet_count = 0;
    tcp_error_count = 0;
    printf("TCP handler initialized successfully\n");
}

void process_tcp_packet(const char* data, int len) {
    if (data == NULL || len <= 0) {
        printf("TCP: Invalid packet data\n");
        tcp_error_count++;
        return;
    }
    
    printf("TCP: Processing packet of length %d bytes\n", len);
    tcp_packet_count++;
    
    // Basic TCP header validation
    if (len < 20) {
        printf("TCP: Packet too small for valid TCP header\n");
        tcp_error_count++;
        return;
    }
    
    printf("TCP: Packet processed successfully (Total: %d)\n", tcp_packet_count);
}

int get_tcp_packet_count() {
    return tcp_packet_count;
}

void print_tcp_statistics() {
    printf("\n=== TCP Statistics ===\n");
    printf("Total TCP packets: %d\n", tcp_packet_count);
    printf("TCP errors: %d\n", tcp_error_count);
    printf("=====================\n");
}

void reset_tcp_counters() {
    printf("Resetting TCP counters...\n");
    tcp_packet_count = 0;
    tcp_error_count = 0;
}