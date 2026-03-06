#include <stdio.h>
#include "icmp_handler.h"
#include "stats.h"

int icmp_packet_count = 0;
int icmp_error_count = 0;

void init_icmp_handler() {
    printf("Initializing ICMP handler...\n");
    icmp_packet_count = 0;
    icmp_error_count = 0;
    printf("ICMP handler initialized successfully\n");
}

void process_icmp_packet(const char* data, int len) {
    if (data == NULL || len <= 0) {
        printf("Invalid ICMP packet data\n");
        icmp_error_count++;
        increment_error_count();
        return;
    }
    
    printf("Processing ICMP packet (length: %d bytes)\n", len);
    icmp_packet_count++;
    increment_packet_count();
    
    // Simulate ICMP packet processing
    printf("ICMP Type: Echo Request/Reply\n");
    printf("ICMP packet processed successfully\n");
}

int get_icmp_packet_count() {
    return icmp_packet_count;
}

void print_icmp_statistics() {
    printf("\n=== ICMP Statistics ===\n");
    printf("Total ICMP packets: %d\n", icmp_packet_count);
    printf("ICMP errors: %d\n", icmp_error_count);
    printf("======================\n");
}

void reset_icmp_handler() {
    printf("Resetting ICMP handler...\n");
    icmp_packet_count = 0;
    icmp_error_count = 0;
    printf("ICMP handler reset complete\n");
}