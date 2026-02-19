#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define NUM_ACCOUNTS 10
#define INITIAL_BALANCE 1000
#define NUM_THREADS 20
#define TOTAL_TRANSACTIONS 1000
#define MAX_TRANSFER_AMOUNT 100

// Global shared data structures - NOT PROTECTED (intentional race conditions)
int accounts[NUM_ACCOUNTS];
int transaction_counter = 0;

typedef struct {
    int thread_id;
    int transactions_to_process;
} thread_data_t;

// Function to get random account index
int get_random_account() {
    return rand() % NUM_ACCOUNTS;
}

// Function to get random transfer amount
int get_random_amount() {
    return (rand() % MAX_TRANSFER_AMOUNT) + 1;
}

// Transaction processing function - CONTAINS RACE CONDITIONS
void* process_transactions(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int transactions = data->transactions_to_process;
    
    for (int i = 0; i < transactions; i++) {
        // Select two different accounts
        int source = get_random_account();
        int destination = get_random_account();
        
        // Ensure source and destination are different
        while (source == destination) {
            destination = get_random_account();
        }
        
        int amount = get_random_amount();
        
        // RACE CONDITION: Reading account balance without lock
        int source_balance = accounts[source];
        
        // Check if source has sufficient funds
        if (source_balance >= amount) {
            // RACE CONDITION: Time window between read and write
            // Multiple threads can read the same balance before any writes occur
            
            // Simulate some processing time to make race condition more likely
            usleep(1);
            
            // RACE CONDITION: Writing to accounts without synchronization
            accounts[source] = source_balance - amount;
            
            // RACE CONDITION: Reading and writing destination account
            int dest_balance = accounts[destination];
            accounts[destination] = dest_balance + amount;
            
            // RACE CONDITION: Incrementing global counter without protection
            transaction_counter++;
        }
    }
    
    free(data);
    return NULL;
}

// Initialize all accounts with initial balance
void initialize_accounts() {
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        accounts[i] = INITIAL_BALANCE;
    }
}

// Calculate total balance across all accounts
int calculate_total_balance() {
    int total = 0;
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        total += accounts[i];
    }
    return total;
}

// Print all account balances
void print_account_balances() {
    printf("\n=== Account Balances ===\n");
    for (int i = 0; i < NUM_ACCOUNTS; i++) {
        printf("Account %2d: $%d\n", i, accounts[i]);
    }
}

int main() {
    pthread_t threads[NUM_THREADS];
    int transactions_per_thread = TOTAL_TRANSACTIONS / NUM_THREADS;
    
    // Seed random number generator
    srand(time(NULL));
    
    // Initialize accounts
    initialize_accounts();
    
    int initial_total = calculate_total_balance();
    printf("Initial Total Balance: $%d\n", initial_total);
    printf("Processing %d transactions across %d threads...\n", 
           TOTAL_TRANSACTIONS, NUM_THREADS);
    
    // Create threads to process transactions
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data_t* data = malloc(sizeof(thread_data_t));
        data->thread_id = i;
        data->transactions_to_process = transactions_per_thread;
        
        if (pthread_create(&threads[i], NULL, process_transactions, data) != 0) {
            fprintf(stderr, "Error creating thread %d\n", i);
            exit(1);
        }
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            fprintf(stderr, "Error joining thread %d\n", i);
            exit(1);
        }
    }
    
    // Print results
    print_account_balances();
    
    int final_total = calculate_total_balance();
    printf("\n=== Transaction Summary ===\n");
    printf("Transactions Processed: %d\n", transaction_counter);
    printf("Final Total Balance: $%d\n", final_total);
    printf("Expected Total Balance: $%d\n", initial_total);
    
    if (final_total != initial_total) {
        printf("ERROR: Balance mismatch! Lost/Gained $%d\n", 
               final_total - initial_total);
    }
    
    return 0;
}