#include "stats.h"
#include <stdio.h>

/* Global statistics counters */
int total_packets = 0;
int total_bytes = 0;
long long total_errors = 0;

/* Initialize all statistics counters to zero */
void init_statistics() {
    total_packets = 0;
    total_bytes = 0;
    total_errors = 0;
    printf("Statistics initialized\n");
}

/* Print current statistics to stdout */
void print_statistics() {
    printf("\n=== Packet Analyzer Statistics ===\n");
    printf("Total Packets: %d\n", total_packets);
    printf("Total Bytes: %d\n", total_bytes);
    printf("Total Errors: %lld\n", total_errors);
    printf("==================================\n\n");
}

/* Reset all statistics counters to zero */
void reset_statistics() {
    total_packets = 0;
    total_bytes = 0;
    total_errors = 0;
    printf("Statistics reset to zero\n");
}

/* Increment the total packet counter */
void increment_total_packets() {
    total_packets++;
}

/* Increment the total bytes counter by specified amount */
void add_bytes(int bytes) {
    if (bytes > 0) {
        total_bytes += bytes;
    }
}

/* Increment the error counter */
void increment_errors() {
    total_errors++;
}

/* Get current packet count */
int get_packet_count() {
    return total_packets;
}

/* Get current byte count */
int get_byte_count() {
    return total_bytes;
}

/* Get current error count */
long long get_error_count() {
    return total_errors;
}