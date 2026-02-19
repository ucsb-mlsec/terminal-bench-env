#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include "account.h"

#define NUM_THREADS 10
#define TRANSACTIONS_PER_THREAD 100
#define NUM_ACCOUNTS 5

typedef struct {
    int thread_id;
    int transactions_completed;
    int transactions_failed;
} ThreadData;

ThreadData thread_stats[NUM_THREADS];
int total_transactions = 0;

void* transaction_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int tid = data->thread_id;
    
    srand(time(NULL) + tid);
    
    for (int i = 0; i < TRANSACTIONS_PER_THREAD; i++) {
        int from_account = rand() % NUM_ACCOUNTS;
        int to_account = rand() % NUM_ACCOUNTS;
        
        while (to_account == from_account) {
            to_account = rand() % NUM_ACCOUNTS;
        }
        
        double amount = (rand() % 100) + 1.0;
        
        // Perform withdrawal
        if (process_transaction(from_account, -amount) == 0) {
            // Perform deposit
            if (process_transaction(to_account, amount) == 0) {
                data->transactions_completed++;
                total_transactions++;
            } else {
                // Rollback withdrawal if deposit fails
                process_transaction(from_account, amount);
                data->transactions_failed++;
            }
        } else {
            data->transactions_failed++;
        }
        
        // Small delay to simulate processing time
        usleep(100);
    }
    
    return NULL;
}

void* deposit_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int tid = data->thread_id;
    
    srand(time(NULL) + tid * 1000);
    
    for (int i = 0; i < TRANSACTIONS_PER_THREAD; i++) {
        int account_id = rand() % NUM_ACCOUNTS;
        double amount = (rand() % 50) + 10.0;
        
        if (process_transaction(account_id, amount) == 0) {
            data->transactions_completed++;
            total_transactions++;
        } else {
            data->transactions_failed++;
        }
        
        usleep(50);
    }
    
    return NULL;
}

void* withdrawal_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int tid = data->thread_id;
    
    srand(time(NULL) + tid * 2000);
    
    for (int i = 0; i < TRANSACTIONS_PER_THREAD; i++) {
        int account_id = rand() % NUM_ACCOUNTS;
        double amount = (rand() % 30) + 5.0;
        
        if (process_transaction(account_id, -amount) == 0) {
            data->transactions_completed++;
            total_transactions++;
        } else {
            data->transactions_failed++;
        }
        
        usleep(50);
    }
    
    return NULL;
}

int run_stress_test() {
    pthread_t threads[NUM_THREADS];
    
    printf("Starting stress test with %d threads...\n", NUM_THREADS);
    printf("Each thread will perform %d transactions\n", TRANSACTIONS_PER_THREAD);
    printf("Total expected transactions: %d\n\n", NUM_THREADS * TRANSACTIONS_PER_THREAD);
    
    // Initialize thread stats
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_stats[i].thread_id = i;
        thread_stats[i].transactions_completed = 0;
        thread_stats[i].transactions_failed = 0;
    }
    
    // Record initial balances
    double initial_total = 0.0;
    printf("Initial account balances:\n");
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        double balance = get_balance(i);
        printf("  Account %d: $%.2f\n", i, balance);
        initial_total += balance;
    }
    printf("Initial total: $%.2f\n\n", initial_total);
    
    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        if (i < NUM_THREADS / 3) {
            pthread_create(&threads[i], NULL, deposit_worker, &thread_stats[i]);
        } else if (i < 2 * NUM_THREADS / 3) {
            pthread_create(&threads[i], NULL, withdrawal_worker, &thread_stats[i]);
        } else {
            pthread_create(&threads[i], NULL, transaction_worker, &thread_stats[i]);
        }
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // Calculate final balances
    double final_total = 0.0;
    printf("\nFinal account balances:\n");
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        double balance = get_balance(i);
        printf("  Account %d: $%.2f\n", i, balance);
        final_total += balance;
    }
    printf("Final total: $%.2f\n\n", final_total);
    
    // Print statistics
    int total_completed = 0;
    int total_failed = 0;
    printf("Thread statistics:\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        printf("  Thread %d: %d completed, %d failed\n", 
               i, thread_stats[i].transactions_completed, 
               thread_stats[i].transactions_failed);
        total_completed += thread_stats[i].transactions_completed;
        total_failed += thread_stats[i].transactions_failed;
    }
    printf("\nTotal completed: %d\n", total_completed);
    printf("Total failed: %d\n", total_failed);
    printf("Total recorded: %d\n", total_transactions);
    
    // Check for balance consistency
    double balance_diff = final_total - initial_total;
    printf("\nBalance difference: $%.2f\n", balance_diff);
    
    if (balance_diff < -0.01 || balance_diff > 0.01) {
        printf("ERROR: Balance mismatch detected! Data corruption occurred.\n");
        return 1;
    }
    
    if (total_transactions != total_completed) {
        printf("ERROR: Transaction count mismatch! Race condition detected.\n");
        return 1;
    }
    
    printf("SUCCESS: All transactions completed correctly!\n");
    return 0;
}

int main(int argc, char* argv[]) {
    printf("Multi-threaded Transaction Service\n");
    printf("===================================\n\n");
    
    // Initialize accounts with starting balance
    init_accounts(NUM_ACCOUNTS, 1000.0);
    
    if (argc > 1 && strcmp(argv[1], "test") == 0) {
        return run_stress_test();
    }
    
    // Default: run a simpler test
    pthread_t threads[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_stats[i].thread_id = i;
        thread_stats[i].transactions_completed = 0;
        thread_stats[i].transactions_failed = 0;
        pthread_create(&threads[i], NULL, transaction_worker, &thread_stats[i]);
    }
    
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("\nFinal balances:\n");
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        printf("Account %d: $%.2f\n", i, get_balance(i));
    }
    
    return 0;
}