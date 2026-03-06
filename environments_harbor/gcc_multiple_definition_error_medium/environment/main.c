#include <stdio.h>
#include "tcp_handler.h"
#include "udp_handler.h"
#include "icmp_handler.h"
#include "stats.h"

int main() {
    printf("Network Packet Analyzer v1.0\n");
    printf("============================\n\n");
    
    // Initialize all protocol handlers
    printf("Initializing protocol handlers...\n");
    init_tcp_handler();
    init_udp_handler();
    init_icmp_handler();
    
    printf("All handlers initialized successfully.\n\n");
    
    // Simulate processing some packets
    printf("Processing TCP packets...\n");
    process_tcp_packet();
    process_tcp_packet();
    
    printf("Processing UDP packets...\n");
    process_udp_packet();
    process_udp_packet();
    process_udp_packet();
    
    printf("Processing ICMP packets...\n");
    process_icmp_packet();
    
    // Display statistics
    printf("\n");
    print_statistics();
    
    // Reset and verify
    printf("\nResetting statistics...\n");
    reset_statistics();
    print_statistics();
    
    printf("\nPacket analyzer shutting down.\n");
    
    return 0;
}