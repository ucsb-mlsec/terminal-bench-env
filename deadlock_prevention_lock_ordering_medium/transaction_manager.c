#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define NUM_ACCOUNTS 5
#define NUM_THREADS 10

// Account structure with balance and associated mutex lock
typedef struct {
    int balance;
    pthread_mutex_t lock;
} Account;

// Global array of accounts
Account accounts[NUM_ACCOUNTS];

// Transfer function - moves money from one account to another
// WARNING: This implementation can cause deadlocks!
void transfer(int from_account, int to_account, int amount) {
    // Lock the from_account first
    pthread_mutex_lock(&accounts[from_account].lock);
    
    // Then lock the to_account
    pthread_mutex_lock(&accounts[to_account].lock);
    
    // Check if sufficient balance exists
    if (accounts[from_account].balance >= amount) {
        // Perform the transfer
        accounts[from_account].balance -= amount;
        accounts[to_account].balance += amount;
        
        printf("Transferred %d from account %d to account %d\n", 
               amount, from_account, to_account);
    } else {
        printf("Insufficient funds in account %d (has %d, needs %d)\n",
               from_account, accounts[from_account].balance, amount);
    }
    
    // Unlock in reverse order
    pthread_mutex_unlock(&accounts[to_account].lock);
    pthread_mutex_unlock(&accounts[from_account].lock);
}

// Worker thread function - performs random transfers
void* worker_thread(void* arg) {
    long thread_id = (long)arg;
    int i;
    
    // Each thread performs 5 transfers
    for (i = 0; i < 5; i++) {
        // Randomly select two different accounts
        int from = rand() % NUM_ACCOUNTS;
        int to = rand() % NUM_ACCOUNTS;
        
        // Ensure from and to are different
        while (from == to) {
            to = rand() % NUM_ACCOUNTS;
        }
        
        // Random transfer amount between 10 and 100
        int amount = (rand() % 91) + 10;
        
        // Perform the transfer
        transfer(from, to, amount);
        
        // Brief pause to simulate processing time
        usleep(1000);
    }
    
    printf("Thread %ld completed all transfers\n", thread_id);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int i;
    long thread_id;
    
    // Seed random number generator
    srand(time(NULL));
    
    // Initialize all accounts
    printf("Initializing accounts...\n");
    for (i = 0; i < NUM_ACCOUNTS; i++) {
        accounts[i].balance = 1000;
        if (pthread_mutex_init(&accounts[i].lock, NULL) != 0) {
            fprintf(stderr, "Failed to initialize mutex for account %d\n", i);
            return 1;
        }
        printf("Account %d: balance = %d\n", i, accounts[i].balance);
    }
    
    printf("\nStarting %d threads...\n\n", NUM_THREADS);
    
    // Create worker threads
    for (thread_id = 0; thread_id < NUM_THREADS; thread_id++) {
        if (pthread_create(&threads[thread_id], NULL, worker_thread, (void*)thread_id) != 0) {
            fprintf(stderr, "Failed to create thread %ld\n", thread_id);
            return 1;
        }
    }
    
    // Wait for all threads to complete
    for (i = 0; i < NUM_THREADS; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "Failed to join thread %d\n", i);
            return 1;
        }
    }
    
    printf("\nAll transfers completed successfully!\n\n");
    
    // Verify total balance and print final account states
    int total_balance = 0;
    printf("Final account balances:\n");
    for (i = 0; i < NUM_ACCOUNTS; i++) {
        printf("Account %d: %d\n", i, accounts[i].balance);
        total_balance += accounts[i].balance;
    }
    
    printf("\nTotal balance across all accounts: %d\n", total_balance);
    printf("Expected total: %d\n", NUM_ACCOUNTS * 1000);
    
    if (total_balance == NUM_ACCOUNTS * 1000) {
        printf("Balance verification PASSED!\n");
    } else {
        printf("Balance verification FAILED!\n");
    }
    
    // Clean up mutexes
    for (i = 0; i < NUM_ACCOUNTS; i++) {
        pthread_mutex_destroy(&accounts[i].lock);
    }
    
    return 0;
}