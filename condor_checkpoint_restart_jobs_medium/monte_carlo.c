#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#define CHECKPOINT_FILE "/tmp/simulation.checkpoint"

volatile sig_atomic_t checkpoint_requested = 0;
long long current_iteration = 0;
long long inside_circle = 0;
long long total_iterations = 0;

void signal_handler(int signum) {
    if (signum == SIGUSR1) {
        checkpoint_requested = 1;
    }
}

void save_checkpoint() {
    FILE *fp = fopen(CHECKPOINT_FILE, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot create checkpoint file\n");
        return;
    }
    fprintf(fp, "%lld\n%lld\n%lld\n", current_iteration, inside_circle, total_iterations);
    fclose(fp);
    checkpoint_requested = 0;
}

int load_checkpoint() {
    FILE *fp = fopen(CHECKPOINT_FILE, "r");
    if (fp == NULL) {
        return 0;
    }
    
    if (fscanf(fp, "%lld\n%lld\n%lld\n", &current_iteration, &inside_circle, &total_iterations) != 3) {
        fclose(fp);
        return 0;
    }
    
    fclose(fp);
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <iterations>\n", argv[0]);
        return 1;
    }
    
    total_iterations = atoll(argv[1]);
    if (total_iterations <= 0) {
        fprintf(stderr, "Error: iterations must be positive\n");
        return 1;
    }
    
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigaction(SIGUSR1, &sa, NULL);
    
    srand(42);
    
    int resumed = load_checkpoint();
    if (resumed) {
        for (long long i = 0; i < current_iteration; i++) {
            double x = (double)rand() / RAND_MAX;
            double y = (double)rand() / RAND_MAX;
        }
    } else {
        current_iteration = 0;
        inside_circle = 0;
    }
    
    for (; current_iteration < total_iterations; current_iteration++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        
        if (x * x + y * y < 1.0) {
            inside_circle++;
        }
        
        if (checkpoint_requested) {
            save_checkpoint();
        }
    }
    
    double pi_estimate = 4.0 * (double)inside_circle / (double)total_iterations;
    printf("%.10f\n", pi_estimate);
    
    unlink(CHECKPOINT_FILE);
    
    return 0;
}