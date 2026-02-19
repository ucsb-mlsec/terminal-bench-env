#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#define VERSION "1.0"

void print_header() {
    printf("========================================\n");
    printf("System Monitor v%s\n", VERSION);
    printf("========================================\n");
}

void print_timestamp() {
    time_t now;
    struct tm *timeinfo;
    char buffer[80];
    
    time(&now);
    timeinfo = localtime(&now);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    printf("Timestamp: %s\n", buffer);
    printf("----------------------------------------\n");
}

double get_simulated_cpu_usage() {
    return 15.5 + (rand() % 50);
}

double get_simulated_memory_usage() {
    return 40.0 + (rand() % 35);
}

double get_simulated_disk_usage() {
    return 55.0 + (rand() % 25);
}

void print_system_metrics() {
    double cpu_usage = get_simulated_cpu_usage();
    double mem_usage = get_simulated_memory_usage();
    double disk_usage = get_simulated_disk_usage();
    
    printf("CPU Status:    %.1f%% utilization\n", cpu_usage);
    printf("Memory Status: %.1f%% used\n", mem_usage);
    printf("Disk Status:   %.1f%% capacity\n", disk_usage);
    printf("----------------------------------------\n");
    
    if (cpu_usage > 80.0) {
        printf("WARNING: High CPU usage detected\n");
    }
    if (mem_usage > 85.0) {
        printf("WARNING: High memory usage detected\n");
    }
    if (disk_usage > 90.0) {
        printf("WARNING: Low disk space\n");
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL) ^ getpid());
    
    print_header();
    print_timestamp();
    print_system_metrics();
    
    printf("Status: All systems operational\n");
    printf("========================================\n");
    
    return 0;
}