#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

/* External declarations for library functions from libcompute.c */
extern double compute_fibonacci(int n);
extern double compute_factorial(int n);
extern double compute_power(double base, int exponent);

/* Global counter to track signal handler invocations */
volatile sig_atomic_t signal_count = 0;
volatile sig_atomic_t failure_count = 0;

/*
 * Signal handler function - tests library safety in asynchronous context
 * This handler calls library functions that perform stack-intensive computations.
 * Without proper compiler flags (like -mno-red-zone on x86_64), the red zone
 * optimization can cause stack corruption when signals interrupt execution.
 */
void signal_handler(int signum) {
    signal_count++;
    
    /* Call library functions that use local variables and stack space */
    double result1 = compute_fibonacci(10);
    double result2 = compute_factorial(8);
    double result3 = compute_power(2.0, 10);
    
    /* Verify results to detect corruption */
    if (result1 != 55.0 || result2 != 40320.0 || result3 != 1024.0) {
        failure_count++;
    }
    
    /* Re-arm the alarm for next iteration */
    if (signal_count < 10) {
        alarm(1);
    }
}

int main() {
    printf("Testing signal handler safety with mathlib...\n");
    printf("This test verifies that library functions can be called safely\n");
    printf("from asynchronous signal handlers without stack corruption.\n\n");
    
    /* Set up signal handler for SIGALRM */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    if (sigaction(SIGALRM, &sa, NULL) == -1) {
        perror("sigaction");
        return 1;
    }
    
    /* Trigger initial alarm */
    alarm(1);
    
    /* Wait for all signal handler invocations to complete */
    while (signal_count < 10) {
        pause(); /* Wait for signals */
    }
    
    printf("\nTest completed:\n");
    printf("  Signal handler invocations: %d\n", signal_count);
    printf("  Result verification failures: %d\n", failure_count);
    
    if (failure_count > 0) {
        printf("\n[FAILED] Stack corruption detected in signal handler!\n");
        return 1;
    } else {
        printf("\n[PASSED] All signal handler calls executed safely.\n");
        return 0;
    }
}