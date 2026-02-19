#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_LINE 256
#define PROC_NET_DEV "/proc/net/dev"

/* Structure to hold network interface statistics */
struct net_stats {
    char interface[32];
    unsigned long rx_bytes;
    unsigned long rx_packets;
    unsigned long rx_errors;
    unsigned long tx_bytes;
    unsigned long tx_packets;
    unsigned long tx_errors;
};

/*
 * Parse a line from /proc/net/dev and extract statistics
 * Returns 0 on success, -1 on failure
 */
int parse_net_line(char *line, struct net_stats *stats) {
    char *colon;
    
    /* Find the colon that separates interface name from stats */
    colon = strchr(line, ':');
    if (!colon) {
        return -1;
    }
    
    /* Extract interface name */
    *colon = '\0';
    sscanf(line, "%s", stats->interface);
    *colon = ':';
    
    /* Parse the statistics */
    if (sscanf(colon + 1, "%lu %lu %lu %*u %*u %*u %*u %*u %lu %lu %lu",
               &stats->rx_bytes, &stats->rx_packets, &stats->rx_errors,
               &stats->tx_bytes, &stats->tx_packets, &stats->tx_errors) != 6) {
        return -1;
    }
    
    return 0;
}

/*
 * Display network statistics for all interfaces
 */
int display_network_stats(void) {
    FILE *fp;
    char line[MAX_LINE];
    struct net_stats stats;
    int line_num = 0;
    
    /* Open /proc/net/dev */
    fp = fopen(PROC_NET_DEV, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", PROC_NET_DEV);
        return -1;
    }
    
    printf("Network Interface Statistics\n");
    printf("========================================\n");
    
    /* Read and process each line */
    while (fgets(line, sizeof(line), fp)) {
        line_num++;
        
        /* Skip header lines */
        if (line_num <= 2) {
            continue;
        }
        
        /* Parse the line */
        if (parse_net_line(line, &stats) == 0) {
            printf("\nInterface: %s\n", stats.interface);
            printf("  RX bytes:   %lu\n", stats.rx_bytes);
            printf("  RX packets: %lu\n", stats.rx_packets);
            printf("  RX errors:  %lu\n", stats.rx_errors);
            printf("  TX bytes:   %lu\n", stats.tx_bytes);
            printf("  TX packets: %lu\n", stats.tx_packets);
            printf("  TX errors:  %lu\n", stats.tx_errors);
        }
    }
    
    fclose(fp);
    return 0;
}

/*
 * Main function - Entry point for the network monitor
 */
int main(int argc, char *argv[]) {
    printf("NetMon - Network Statistics Monitor\n\n");
    
    /* Display current network statistics */
    if (display_network_stats() != 0) {
        fprintf(stderr, "Error: Failed to read network statistics\n");
        return EXIT_FAILURE;
    }
    
    printf("\n");
    return EXIT_SUCCESS;
}