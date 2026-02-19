/*
 * System Monitor v1.0
 * A simple system monitoring application for embedded Linux
 * 
 * This application provides basic system monitoring capabilities
 * including uptime, memory usage, and CPU load statistics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#define VERSION "1.0"
#define DEFAULT_INTERVAL 5

/* Structure to hold monitoring data */
typedef struct {
    time_t current_time;
    long uptime;
    int memory_percent;
    int cpu_percent;
} monitor_data_t;

/*
 * Display usage information
 */
void print_usage(const char *program_name) {
    printf("System Monitor v%s\n", VERSION);
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Options:\n");
    printf("  -h, --help        Show this help message\n");
    printf("  -d, --daemon      Run continuously in daemon mode\n");
    printf("  -i <seconds>      Interval between checks (default: %d seconds)\n", DEFAULT_INTERVAL);
    printf("\nExample:\n");
    printf("  %s -d -i 10       Run as daemon with 10 second interval\n", program_name);
}

/*
 * Collect system monitoring data
 * In a real implementation, this would read from /proc filesystem
 * For this embedded system, we use simulated data
 */
void collect_system_data(monitor_data_t *data) {
    static long base_uptime = 0;
    static int initialized = 0;
    
    /* Get current time */
    data->current_time = time(NULL);
    
    /* Simulate uptime (incrementing counter) */
    if (!initialized) {
        base_uptime = 1000 + (rand() % 5000);
        initialized = 1;
    }
    data->uptime = base_uptime++;
    
    /* Simulate memory usage (30-70%) */
    data->memory_percent = 30 + (rand() % 40);
    
    /* Simulate CPU load (10-50%) */
    data->cpu_percent = 10 + (rand() % 40);
}

/*
 * Display monitoring information
 */
void display_monitor_data(const monitor_data_t *data) {
    char time_str[64];
    struct tm *tm_info;
    
    /* Format time string */
    tm_info = localtime(&data->current_time);
    if (tm_info != NULL) {
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);
    } else {
        snprintf(time_str, sizeof(time_str), "N/A");
    }
    
    /* Display monitoring data */
    printf("\n=== System Monitor ===\n");
    printf("Time: %s\n", time_str);
    printf("Uptime: %ld seconds\n", data->uptime);
    printf("Memory: %d%%\n", data->memory_percent);
    printf("CPU: %d%%\n", data->cpu_percent);
    printf("======================\n");
}

/*
 * Main monitoring loop
 */
int run_monitor(int daemon_mode, int interval) {
    monitor_data_t data;
    
    printf("System Monitor v%s\n", VERSION);
    if (daemon_mode) {
        printf("Running in daemon mode (interval: %d seconds)\n", interval);
        printf("Press Ctrl+C to stop\n");
    }
    
    /* Initialize random seed for simulated data */
    srand(time(NULL));
    
    do {
        /* Collect and display system data */
        collect_system_data(&data);
        display_monitor_data(&data);
        
        /* Sleep if in daemon mode */
        if (daemon_mode) {
            sleep(interval);
        }
        
    } while (daemon_mode);
    
    return 0;
}

/*
 * Main entry point
 */
int main(int argc, char *argv[]) {
    int daemon_mode = 0;
    int interval = DEFAULT_INTERVAL;
    int i;
    
    /* Parse command-line arguments */
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--daemon") == 0) {
            daemon_mode = 1;
        }
        else if (strcmp(argv[i], "-i") == 0) {
            /* Check if next argument exists */
            if (i + 1 < argc) {
                interval = atoi(argv[i + 1]);
                if (interval <= 0) {
                    fprintf(stderr, "Error: Invalid interval value: %s\n", argv[i + 1]);
                    fprintf(stderr, "Interval must be a positive integer\n");
                    return 1;
                }
                i++; /* Skip next argument */
            } else {
                fprintf(stderr, "Error: -i option requires an argument\n");
                print_usage(argv[0]);
                return 1;
            }
        }
        else {
            fprintf(stderr, "Error: Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    /* Run the monitor */
    return run_monitor(daemon_mode, interval);
}